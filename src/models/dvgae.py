"""
Deep Variational Graph Autoencoder (DVGAE) for Knowledge Graph Embeddings.

Implements a variational graph autoencoder that:
1. Encodes graph structure into low-dimensional node embeddings
2. Preserves directional relationships (critical for knowledge graphs)
3. Enables link prediction for novel drug-gene-disease associations

Based on the CardioKG methodology from Imperial College London.
Reference: Kipf & Welling (2016) "Variational Graph Auto-Encoders"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, train_test_split_edges
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network Encoder.
    
    Uses multiple GCN layers to aggregate neighbor information,
    producing mean and log-variance for the variational component.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layers for mean and log-variance
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            mu: Mean of latent distribution
            logstd: Log standard deviation of latent distribution
        """
        # Pass through GCN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Compute mean and log-std
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd


class GATEncoder(nn.Module):
    """
    Graph Attention Network Encoder.
    
    Uses attention mechanisms to weight neighbor contributions,
    often better for heterogeneous knowledge graphs.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.dropout = dropout
        
        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # Output layers
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        
        return mu, logstd


class InnerProductDecoder(nn.Module):
    """
    Inner product decoder for link prediction.
    
    Predicts edge probability as sigmoid(z_i^T * z_j)
    """
    
    def forward(
        self, 
        z: torch.Tensor, 
        edge_index: torch.Tensor,
        sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Decode edge probabilities.
        
        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edges to predict [2, num_edges]
            sigmoid: Whether to apply sigmoid
            
        Returns:
            Edge probabilities
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class DirectionalDecoder(nn.Module):
    """
    Directional decoder that preserves edge direction.
    
    Uses separate transformations for source and target nodes
    to capture asymmetric relationships (e.g., Drug TARGETS Gene).
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.source_transform = nn.Linear(embedding_dim, hidden_dim)
        self.target_transform = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        z: torch.Tensor, 
        edge_index: torch.Tensor,
        sigmoid: bool = True
    ) -> torch.Tensor:
        """
        Decode edge probabilities with direction awareness.
        """
        source_z = self.source_transform(z[edge_index[0]])
        target_z = self.target_transform(z[edge_index[1]])
        
        # Combine and predict
        combined = F.relu(source_z + target_z)
        value = self.output(combined).squeeze()
        
        return torch.sigmoid(value) if sigmoid else value


class DVGAE(nn.Module):
    """
    Deep Variational Graph Autoencoder.
    
    Combines encoder and decoder for unsupervised graph representation learning.
    Generates node embeddings that capture graph structure for downstream
    link prediction tasks.
    
    Key features:
    - Variational inference for robust embeddings
    - Multiple encoder architectures (GCN, GAT)
    - Directional decoder for knowledge graphs
    - Support for node type features
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        encoder_type: str = 'gcn',
        directional: bool = True,
        num_layers: int = 2,
        dropout: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize DVGAE.
        
        Args:
            num_nodes: Number of nodes in graph
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Embedding dimension
            encoder_type: 'gcn' or 'gat'
            directional: Use directional decoder
            num_layers: Number of encoder layers
            dropout: Dropout rate
            device: Device to use
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.device = device
        
        # Build encoder
        if encoder_type == 'gcn':
            self.encoder = GCNEncoder(
                in_channels, hidden_channels, out_channels,
                num_layers=num_layers, dropout=dropout
            )
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(
                in_channels, hidden_channels, out_channels,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Build decoder
        if directional:
            self.decoder = DirectionalDecoder(out_channels)
        else:
            self.decoder = InnerProductDecoder()
        
        self.to(device)
        
    def encode(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode graph into latent space."""
        return self.encoder(x, edge_index)
    
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        Reparametrization trick for variational inference.
        z = mu + std * epsilon, where epsilon ~ N(0, 1)
        """
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def decode(
        self, 
        z: torch.Tensor, 
        edge_index: torch.Tensor,
        sigmoid: bool = True
    ) -> torch.Tensor:
        """Decode edge probabilities from embeddings."""
        return self.decoder(z, edge_index, sigmoid=sigmoid)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            z: Node embeddings
            mu: Mean of latent distribution
            logstd: Log standard deviation
        """
        mu, logstd = self.encode(x, edge_index)
        z = self.reparametrize(mu, logstd)
        return z, mu, logstd
    
    def kl_loss(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """
        KL divergence loss: KL(q(z|x) || p(z))
        where p(z) = N(0, I)
        """
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        )
    
    def recon_loss(
        self, 
        z: torch.Tensor, 
        pos_edge_index: torch.Tensor,
        neg_edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reconstruction loss using positive and negative edges.
        
        Args:
            z: Node embeddings
            pos_edge_index: Positive (existing) edges
            neg_edge_index: Negative (non-existing) edges
        """
        # Positive edge loss
        pos_loss = -torch.log(
            self.decode(z, pos_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        # Negative edge loss
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                pos_edge_index, 
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        
        neg_loss = -torch.log(
            1 - self.decode(z, neg_edge_index, sigmoid=True) + 1e-15
        ).mean()
        
        return pos_loss + neg_loss
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Get node embeddings as numpy array."""
        self.eval()
        with torch.no_grad():
            z, _, _ = self.forward(x, edge_index)
            return z.cpu().numpy()


class DVGAETrainer:
    """
    Trainer for DVGAE model.
    
    Handles training loop, evaluation, and embedding extraction
    for the knowledge graph pipeline.
    """
    
    def __init__(
        self,
        model: DVGAE,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        kl_weight: float = 1.0
    ):
        self.model = model
        self.kl_weight = kl_weight
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.val_aucs = []
        
    def train_epoch(
        self, 
        data: Data,
        neg_edge_index: Optional[torch.Tensor] = None
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        z, mu, logstd = self.model(data.x, data.edge_index)
        
        # Compute losses
        recon_loss = self.model.recon_loss(z, data.edge_index, neg_edge_index)
        kl_loss = self.model.kl_loss(mu, logstd)
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    @torch.no_grad()
    def evaluate(
        self, 
        data: Data,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Evaluate model using AUC and AP metrics.
        
        Returns:
            auc: Area under ROC curve
            ap: Average precision
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        self.model.eval()
        
        z, _, _ = self.model(data.x, data.edge_index)
        
        # Get predictions
        pos_pred = self.model.decode(z, pos_edge_index, sigmoid=True)
        neg_pred = self.model.decode(z, neg_edge_index, sigmoid=True)
        
        # Combine predictions and labels
        preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ]).numpy()
        
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        
        return auc, ap
    
    def fit(
        self,
        data: Data,
        epochs: int = 200,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        early_stopping: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with validation.
        
        Args:
            data: PyTorch Geometric Data object
            epochs: Number of training epochs
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            early_stopping: Patience for early stopping
            verbose: Print progress
            
        Returns:
            Dictionary with training results
        """
        # Split edges for train/val/test
        data = train_test_split_edges(data, val_ratio, test_ratio)
        
        device = self.model.device
        data = data.to(device)
        
        best_val_auc = 0
        patience_counter = 0
        best_embeddings = None
        
        for epoch in range(epochs):
            # Train
            loss = self.train_epoch(data)
            self.train_losses.append(loss)
            
            # Validate
            val_auc, val_ap = self.evaluate(
                data, 
                data.val_pos_edge_index,
                data.val_neg_edge_index
            )
            self.val_aucs.append(val_auc)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_embeddings = self.model.get_embeddings(data.x, data.train_pos_edge_index)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final test evaluation
        test_auc, test_ap = self.evaluate(
            data,
            data.test_pos_edge_index,
            data.test_neg_edge_index
        )
        
        logger.info(f"Final Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_aucs': self.val_aucs,
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'embeddings': best_embeddings
        }
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        node_ids: List[str],
        output_path: str = 'embeddings/dvgae_embeddings.pkl'
    ):
        """Save embeddings with node IDs."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        embedding_dict = {node_id: embeddings[i] for i, node_id in enumerate(node_ids)}
        
        with open(output_path, 'wb') as f:
            pickle.dump(embedding_dict, f)
        
        logger.info(f"Saved {len(embedding_dict)} embeddings to {output_path}")


def networkx_to_pyg(
    G: nx.Graph,
    node_features: Optional[Dict[str, np.ndarray]] = None,
    feature_dim: int = 64
) -> Tuple[Data, List[str]]:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph
        node_features: Optional pre-computed node features
        feature_dim: Dimension for random features if not provided
        
    Returns:
        data: PyTorch Geometric Data object
        node_ids: List of node IDs in order
    """
    # Create node ID to index mapping
    node_ids = list(G.nodes())
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Create edge index
    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([
            [node_id_to_idx[u], node_id_to_idx[v]]
            for u, v in edges
        ], dtype=torch.long).t().contiguous()
    
    # Create node features
    num_nodes = len(node_ids)
    if node_features is not None:
        x = torch.tensor([
            node_features.get(node_id, np.random.randn(feature_dim))
            for node_id in node_ids
        ], dtype=torch.float)
    else:
        # Create features based on node type (one-hot) + random
        node_types = set()
        for node_id in node_ids:
            nt = G.nodes[node_id].get('node_type', 'Unknown')
            node_types.add(nt)
        
        type_to_idx = {t: i for i, t in enumerate(sorted(node_types))}
        num_types = len(node_types)
        
        x = torch.zeros((num_nodes, num_types + feature_dim))
        for idx, node_id in enumerate(node_ids):
            nt = G.nodes[node_id].get('node_type', 'Unknown')
            x[idx, type_to_idx[nt]] = 1.0
            x[idx, num_types:] = torch.randn(feature_dim) * 0.1
    
    data = Data(x=x, edge_index=edge_index)
    
    return data, node_ids


def train_dvgae_on_graph(
    G: nx.Graph,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 0.01,
    encoder_type: str = 'gcn',
    device: str = None,
    save_path: str = 'embeddings/dvgae_embeddings.pkl'
) -> Dict:
    """
    Convenience function to train DVGAE on a NetworkX graph.
    
    Args:
        G: NetworkX graph
        embedding_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        epochs: Training epochs
        learning_rate: Learning rate
        encoder_type: 'gcn' or 'gat'
        device: Device to use (auto-detected if None)
        save_path: Path to save embeddings
        
    Returns:
        Dictionary with results including embeddings
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Training DVGAE on device: {device}")
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to PyG format
    data, node_ids = networkx_to_pyg(G)
    
    # Initialize model
    model = DVGAE(
        num_nodes=len(node_ids),
        in_channels=data.x.size(1),
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        encoder_type=encoder_type,
        directional=True,
        device=device
    )
    
    # Train
    trainer = DVGAETrainer(model, learning_rate=learning_rate)
    results = trainer.fit(data, epochs=epochs)
    
    # Save embeddings
    trainer.save_embeddings(results['embeddings'], node_ids, save_path)
    
    # Add node_ids to results
    results['node_ids'] = node_ids
    
    return results


def main():
    """Demo DVGAE training on sample graph."""
    # Create sample knowledge graph
    G = nx.DiGraph()
    
    # Add nodes with types
    for i in range(10):
        G.add_node(f"Drug_{i}", node_type='Drug')
    for i in range(20):
        G.add_node(f"Gene_{i}", node_type='Gene')
    for i in range(5):
        G.add_node(f"Disease_{i}", node_type='Disease')
    
    # Add random edges
    import random
    random.seed(42)
    
    for i in range(10):
        gene = f"Gene_{random.randint(0, 19)}"
        G.add_edge(f"Drug_{i}", gene, relationship='TARGETS')
    
    for i in range(20):
        disease = f"Disease_{random.randint(0, 4)}"
        G.add_edge(f"Gene_{i}", disease, relationship='ASSOCIATED_WITH')
    
    for i in range(15):
        g1 = f"Gene_{random.randint(0, 19)}"
        g2 = f"Gene_{random.randint(0, 19)}"
        if g1 != g2:
            G.add_edge(g1, g2, relationship='INTERACTS_WITH')
    
    logger.info("Training DVGAE on sample graph...")
    results = train_dvgae_on_graph(
        G,
        embedding_dim=32,
        hidden_dim=64,
        epochs=100,
        save_path='embeddings/demo_dvgae_embeddings.pkl'
    )
    
    logger.info(f"Best validation AUC: {results['best_val_auc']:.4f}")
    logger.info(f"Test AUC: {results['test_auc']:.4f}")


if __name__ == '__main__':
    main()

