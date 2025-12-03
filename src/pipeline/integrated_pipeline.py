"""
Integrated GNN and Visualization Pipeline for Cardiotoxicity Knowledge Graph.

This module integrates the complete workflow:
1. Visualization Phase: Force-directed layouts reveal structure
2. GNN Phase: DVGAE generates embeddings via message passing
3. Prediction Phase: Link prediction for drug-gene-disease associations
4. Interpretation Phase: Path tracing explains predictions

The pipeline follows the methodology from CardioKG (Imperial College London),
adapting it for cardiotoxicity research applications.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.graph_visualizer import GraphVisualizer
from models.dvgae import DVGAE, DVGAETrainer, networkx_to_pyg, train_dvgae_on_graph

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    import torch
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, 
        classification_report, confusion_matrix
    )
    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the integrated pipeline."""
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "output"
    
    # Neo4j connection (optional)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # GNN configuration
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    epochs: int = 200
    learning_rate: float = 0.01
    encoder_type: str = "gcn"  # "gcn" or "gat"
    
    # Visualization
    layout_algorithm: str = "spring"
    community_method: str = "louvain"
    
    # Link prediction
    classifier: str = "random_forest"
    test_size: float = 0.2
    negative_ratio: float = 1.0
    
    # Device
    device: str = "auto"


@dataclass
class PredictionResult:
    """Result of a link prediction."""
    source: str
    target: str
    source_type: str
    target_type: str
    probability: float
    paths: List[List[str]] = field(default_factory=list)
    mechanism: str = ""
    confidence: str = "low"
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'source_type': self.source_type,
            'target_type': self.target_type,
            'probability': self.probability,
            'confidence': self.confidence,
            'mechanism': self.mechanism,
            'paths': self.paths
        }


class CardioKGPipeline:
    """
    Integrated Knowledge Graph Pipeline for Cardiotoxicity Analysis.
    
    Workflow:
    1. Load/Build Graph → Data from Neo4j or CSV files
    2. Visualize Structure → Force-directed layout, community detection
    3. Train GNN → DVGAE for node embeddings
    4. Link Prediction → Classify novel associations
    5. Interpret Results → Path tracing for mechanistic explanation
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize pipeline with configuration."""
        self.config = config or PipelineConfig()
        
        # Set device
        if self.config.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Initialize components
        self.graph: nx.DiGraph = None
        self.visualizer: GraphVisualizer = None
        self.embeddings: Dict[str, np.ndarray] = None
        self.link_predictor = None
        
        # Tracking
        self.node_types: Dict[str, str] = {}
        self.edge_types: Dict[Tuple[str, str], str] = {}
        
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized (device: {self.device})")
    
    # ===== Phase 1: Data Loading =====
    
    def load_graph_from_neo4j(self) -> nx.DiGraph:
        """
        Load knowledge graph from Neo4j database.
        
        Returns:
            NetworkX directed graph
        """
        if not HAS_NEO4J:
            raise ImportError("neo4j package required. Install with: pip install neo4j")
        
        logger.info("Loading graph from Neo4j...")
        
        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        G = nx.DiGraph()
        
        with driver.session() as session:
            # Get all nodes
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, properties(n) as props, id(n) as id
            """)
            
            for record in result:
                labels = record['labels']
                props = record['props']
                node_id = props.get('name') or props.get('id') or str(record['id'])
                node_type = labels[0] if labels else 'Unknown'
                
                G.add_node(node_id, node_type=node_type, **props)
                self.node_types[node_id] = node_type
            
            # Get all relationships
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN 
                    coalesce(a.name, toString(id(a))) as source,
                    coalesce(b.name, toString(id(b))) as target,
                    type(r) as rel_type,
                    properties(r) as props
            """)
            
            for record in result:
                source = record['source']
                target = record['target']
                rel_type = record['rel_type']
                props = record['props'] or {}
                
                G.add_edge(source, target, relationship=rel_type, **props)
                self.edge_types[(source, target)] = rel_type
        
        driver.close()
        
        self.graph = G
        self.visualizer = GraphVisualizer(G)
        
        logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def load_graph_from_csv(
        self,
        drug_protein_path: str = None,
        pathway_path: str = None,
        ppi_path: str = None,
        gene_disease_path: str = None
    ) -> nx.DiGraph:
        """
        Load knowledge graph from CSV files.
        
        Args:
            drug_protein_path: Path to drug-protein targets CSV
            pathway_path: Path to protein-pathway CSV
            ppi_path: Path to protein-protein interactions CSV
            gene_disease_path: Path to gene-disease associations CSV
            
        Returns:
            NetworkX directed graph
        """
        logger.info("Building graph from CSV files...")
        
        G = nx.DiGraph()
        raw_dir = Path(self.config.raw_data_dir)
        
        # Default paths
        drug_protein_path = drug_protein_path or raw_dir / "clean_drug_protein_targets.csv"
        pathway_path = pathway_path or raw_dir / "Pathway_Cardiovascular_Filtered.csv"
        
        # Load drug-protein targets
        if Path(drug_protein_path).exists():
            df = pd.read_csv(drug_protein_path)
            logger.info(f"Loading {len(df)} drug-protein relationships...")
            
            for _, row in df.iterrows():
                drug_id = str(row.get('drugbank_id', '')).strip()
                drug_name = str(row.get('drug_name', drug_id)).strip()
                protein_id = str(row.get('uniprot_id', '')).strip()
                protein_name = str(row.get('uniprot_name', protein_id)).strip()
                
                if drug_id and protein_id:
                    # Add drug node
                    G.add_node(drug_name, node_type='Drug', drugbank_id=drug_id)
                    self.node_types[drug_name] = 'Drug'
                    
                    # Add protein node
                    G.add_node(protein_id, node_type='Protein', name=protein_name)
                    self.node_types[protein_id] = 'Protein'
                    
                    # Add edge
                    G.add_edge(drug_name, protein_id, relationship='TARGETS')
                    self.edge_types[(drug_name, protein_id)] = 'TARGETS'
        
        # Load pathway data
        if Path(pathway_path).exists():
            df = pd.read_csv(pathway_path)
            logger.info(f"Loading {len(df)} protein-pathway relationships...")
            
            for _, row in df.iterrows():
                protein_id = str(row.get('uniprot_id', '')).strip()
                # Handle potential prefix in the data
                if ':' in protein_id:
                    protein_id = protein_id.split(':')[-1].strip()
                
                pathway_id = str(row.get('reactome_pathway_id', '')).strip()
                pathway_name = str(row.get('pathway_name', pathway_id)).strip()
                
                if protein_id and pathway_id:
                    # Add/update protein node
                    if protein_id not in G:
                        G.add_node(protein_id, node_type='Protein')
                        self.node_types[protein_id] = 'Protein'
                    
                    # Add pathway node
                    G.add_node(pathway_name, node_type='Pathway', pathway_id=pathway_id)
                    self.node_types[pathway_name] = 'Pathway'
                    
                    # Add edge
                    G.add_edge(protein_id, pathway_name, relationship='INVOLVED_IN')
                    self.edge_types[(protein_id, pathway_name)] = 'INVOLVED_IN'
        
        # Load PPI data if available
        if ppi_path and Path(ppi_path).exists():
            df = pd.read_csv(ppi_path)
            logger.info(f"Loading {len(df)} protein-protein interactions...")
            
            # Create mapping from gene symbol/uniprot to existing protein nodes
            existing_proteins = {n for n, t in self.node_types.items() if t == 'Protein'}
            
            ppi_added = 0
            for _, row in df.iterrows():
                # Try multiple columns for protein identifiers
                protein1 = str(row.get('uniprot_a', row.get('protein1', row.get('source', '')))).strip()
                protein2 = str(row.get('uniprot_b', row.get('protein2', row.get('target', '')))).strip()
                
                # Also try gene symbols
                gene1 = str(row.get('gene_a', '')).strip()
                gene2 = str(row.get('gene_b', '')).strip()
                
                score = row.get('combined_score', row.get('score', 0.5))
                if score == '-' or pd.isna(score):
                    score = 0.5
                
                # Use UniProt ID if available, otherwise gene symbol
                p1 = protein1 if protein1 and protein1 != 'nan' else gene1
                p2 = protein2 if protein2 and protein2 != 'nan' else gene2
                
                if p1 and p2 and p1 != p2:
                    # Add nodes if they overlap with existing proteins or are new
                    if p1 in existing_proteins or p1 not in G:
                        if p1 not in G:
                            G.add_node(p1, node_type='Protein')
                            self.node_types[p1] = 'Protein'
                    if p2 in existing_proteins or p2 not in G:
                        if p2 not in G:
                            G.add_node(p2, node_type='Protein')
                            self.node_types[p2] = 'Protein'
                    
                    # Only add edge if both nodes are now in graph
                    if p1 in G and p2 in G:
                        G.add_edge(p1, p2, relationship='INTERACTS_WITH', score=score)
                        self.edge_types[(p1, p2)] = 'INTERACTS_WITH'
                        ppi_added += 1
                        
                        # Limit for manageable graph size
                        if ppi_added >= 50000:
                            logger.info(f"Limiting PPI edges to {ppi_added} for manageability")
                            break
            
            logger.info(f"Added {ppi_added} PPI edges to graph")
        
        # Load gene-disease data if available
        if gene_disease_path and Path(gene_disease_path).exists():
            df = pd.read_csv(gene_disease_path, sep='\t' if gene_disease_path.endswith('.tsv') else ',')
            logger.info(f"Loading {len(df)} gene-disease associations...")
            
            for _, row in df.iterrows():
                gene = str(row.get('geneSymbol', row.get('gene', ''))).strip()
                disease = str(row.get('diseaseName', row.get('disease', ''))).strip()
                score = row.get('score', 0.5)
                
                if gene and disease:
                    if gene not in G:
                        G.add_node(gene, node_type='Gene')
                        self.node_types[gene] = 'Gene'
                    
                    G.add_node(disease, node_type='Disease')
                    self.node_types[disease] = 'Disease'
                    
                    G.add_edge(gene, disease, relationship='ASSOCIATED_WITH', score=score)
                    self.edge_types[(gene, disease)] = 'ASSOCIATED_WITH'
        
        self.graph = G
        self.visualizer = GraphVisualizer(G)
        
        logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def set_graph(self, G: nx.DiGraph):
        """Set a custom graph."""
        self.graph = G
        self.visualizer = GraphVisualizer(G)
        
        # Extract node types
        for node in G.nodes():
            self.node_types[node] = G.nodes[node].get('node_type', 'Unknown')
        
        # Extract edge types
        for u, v in G.edges():
            self.edge_types[(u, v)] = G.edges[u, v].get('relationship', 'Unknown')
        
        logger.info(f"Graph set: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # ===== Phase 2: Visualization Analysis =====
    
    def analyze_structure(
        self,
        save_plots: bool = True,
        output_prefix: str = "structure"
    ) -> Dict[str, Any]:
        """
        Perform structural analysis using force-directed layouts.
        
        This phase reveals:
        - Hub nodes (highly connected entities)
        - Community structure (clusters)
        - Connectivity patterns
        - Potential data quality issues
        
        Returns:
            Dictionary with analysis results
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Call load_graph_from_* first.")
        
        logger.info("🔍 Phase 2: Structural Analysis...")
        
        results = {}
        
        # Compute layout
        pos = self.visualizer.compute_force_directed_layout(
            algorithm=self.config.layout_algorithm
        )
        results['layout'] = pos
        
        # Detect communities
        communities = self.visualizer.detect_communities(
            method=self.config.community_method
        )
        results['communities'] = communities
        results['num_communities'] = len(set(communities.values()))
        
        # Identify hubs
        hub_metrics = ['degree', 'betweenness', 'pagerank']
        results['hubs'] = {}
        
        for metric in hub_metrics:
            try:
                hubs = self.visualizer.identify_hub_nodes(metric=metric, top_k=10)
                results['hubs'][metric] = hubs
            except Exception as e:
                logger.warning(f"Could not compute {metric} centrality: {e}")
        
        # Graph statistics
        results['statistics'] = self.visualizer.get_graph_statistics()
        
        # Determine optimal GNN depth based on graph structure
        avg_path_length = self._estimate_path_length()
        results['recommended_gnn_layers'] = min(max(2, int(avg_path_length)), 4)
        
        # Save visualizations
        if save_plots:
            output_dir = Path(self.config.output_dir)
            
            # Main graph visualization
            self.visualizer.plot_graph(
                pos=pos,
                color_by='node_type',
                title='Knowledge Graph Structure',
                save_path=str(output_dir / f"{output_prefix}_graph.png")
            )
            
            # Community visualization
            self.visualizer.plot_graph(
                pos=pos,
                color_by='community',
                title='Community Structure',
                save_path=str(output_dir / f"{output_prefix}_communities.png")
            )
            
            # Degree distribution
            self.visualizer.plot_degree_distribution(
                save_path=str(output_dir / f"{output_prefix}_degree_dist.png")
            )
        
        logger.info(f"Found {results['num_communities']} communities")
        logger.info(f"Recommended GNN layers: {results['recommended_gnn_layers']}")
        
        return results
    
    def _estimate_path_length(self) -> float:
        """Estimate average shortest path length for GNN depth recommendation."""
        try:
            G_undirected = self.graph.to_undirected()
            if nx.is_connected(G_undirected):
                return nx.average_shortest_path_length(G_undirected)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                subgraph = G_undirected.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except:
            return 3.0  # Default
    
    # ===== Phase 3: GNN Embedding =====
    
    def train_embeddings(
        self,
        epochs: int = None,
        save_path: str = None
    ) -> Dict:
        """
        Train GNN (DVGAE) to generate node embeddings.
        
        The GNN learns by:
        1. Message passing between neighbors
        2. Aggregating neighborhood information
        3. Encoding into low-dimensional embeddings
        
        Returns:
            Training results including embeddings
        """
        if self.graph is None:
            raise ValueError("No graph loaded.")
        
        logger.info("🧠 Phase 3: GNN Embedding Training...")
        
        epochs = epochs or self.config.epochs
        save_path = save_path or str(Path(self.config.output_dir) / "dvgae_embeddings.pkl")
        
        # Train DVGAE
        results = train_dvgae_on_graph(
            self.graph,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            epochs=epochs,
            learning_rate=self.config.learning_rate,
            encoder_type=self.config.encoder_type,
            device=self.device,
            save_path=save_path
        )
        
        # Store embeddings
        node_ids = results['node_ids']
        embeddings_array = results['embeddings']
        
        self.embeddings = {
            node_id: embeddings_array[i]
            for i, node_id in enumerate(node_ids)
        }
        
        logger.info(f"Generated embeddings for {len(self.embeddings)} nodes")
        logger.info(f"Test AUC: {results['test_auc']:.4f}")
        
        return results
    
    def load_embeddings(self, path: str):
        """Load pre-trained embeddings."""
        with open(path, 'rb') as f:
            self.embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings for {len(self.embeddings)} nodes")
    
    # ===== Phase 4: Link Prediction =====
    
    def train_link_predictor(
        self,
        positive_edges: List[Tuple[str, str]] = None,
        negative_ratio: float = None
    ) -> Dict:
        """
        Train classifier for link prediction.
        
        Uses concatenated node embeddings to predict
        whether an edge should exist between two nodes.
        
        Args:
            positive_edges: Known edges (uses graph edges if None)
            negative_ratio: Ratio of negative to positive samples
            
        Returns:
            Training results and metrics
        """
        if self.embeddings is None:
            raise ValueError("No embeddings. Call train_embeddings first.")
        
        if not HAS_ML:
            raise ImportError("sklearn required for link prediction")
        
        logger.info("🎯 Phase 4: Link Prediction Training...")
        
        negative_ratio = negative_ratio or self.config.negative_ratio
        
        # Get positive edges
        if positive_edges is None:
            positive_edges = list(self.graph.edges())
        
        # Filter to edges where both nodes have embeddings
        positive_edges = [
            (u, v) for u, v in positive_edges
            if u in self.embeddings and v in self.embeddings
        ]
        
        # Generate negative edges
        all_nodes = list(self.embeddings.keys())
        existing_edges = set(positive_edges)
        
        num_neg = int(len(positive_edges) * negative_ratio)
        negative_edges = []
        
        np.random.seed(42)
        while len(negative_edges) < num_neg:
            u = np.random.choice(all_nodes)
            v = np.random.choice(all_nodes)
            if u != v and (u, v) not in existing_edges:
                negative_edges.append((u, v))
        
        # Create feature matrix
        X_pos = np.array([
            np.concatenate([self.embeddings[u], self.embeddings[v]])
            for u, v in positive_edges
        ])
        
        X_neg = np.array([
            np.concatenate([self.embeddings[u], self.embeddings[v]])
            for u, v in negative_edges
        ])
        
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * len(positive_edges) + [0] * len(negative_edges))
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42, stratify=y
        )
        
        # Train classifier
        if self.config.classifier == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        clf.fit(X_train, y_train)
        self.link_predictor = clf
        
        # Evaluate
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'ap': average_precision_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Link Prediction AUC: {results['auc']:.4f}")
        logger.info(f"Link Prediction AP: {results['ap']:.4f}")
        
        return results
    
    def predict_links(
        self,
        source_type: str = 'Drug',
        target_type: str = 'Disease',
        top_k: int = 20,
        min_probability: float = 0.5
    ) -> List[PredictionResult]:
        """
        Predict novel links between node types.
        
        Args:
            source_type: Source node type (e.g., 'Drug')
            target_type: Target node type (e.g., 'Disease')
            top_k: Number of top predictions to return
            min_probability: Minimum probability threshold
            
        Returns:
            List of PredictionResult objects
        """
        if self.link_predictor is None:
            raise ValueError("No link predictor. Call train_link_predictor first.")
        
        logger.info(f"Predicting {source_type} → {target_type} links...")
        
        # Get nodes of each type
        source_nodes = [n for n, t in self.node_types.items() if t == source_type]
        target_nodes = [n for n, t in self.node_types.items() if t == target_type]
        
        # Filter to nodes with embeddings
        source_nodes = [n for n in source_nodes if n in self.embeddings]
        target_nodes = [n for n in target_nodes if n in self.embeddings]
        
        # Existing edges
        existing_edges = set(self.graph.edges())
        
        # Generate predictions for all pairs
        predictions = []
        
        for source in source_nodes:
            for target in target_nodes:
                if (source, target) not in existing_edges:
                    # Create feature vector
                    x = np.concatenate([
                        self.embeddings[source],
                        self.embeddings[target]
                    ]).reshape(1, -1)
                    
                    prob = self.link_predictor.predict_proba(x)[0, 1]
                    
                    if prob >= min_probability:
                        predictions.append((source, target, prob))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[2], reverse=True)
        predictions = predictions[:top_k]
        
        # Create detailed results with path tracing
        results = []
        for source, target, prob in predictions:
            # Trace paths for interpretation
            paths = self.visualizer.trace_path(source, target, max_length=4)
            
            # Generate mechanism description
            mechanism = self._generate_mechanism_description(source, target, paths)
            
            # Determine confidence
            if prob > 0.8 and len(paths) > 0:
                confidence = 'high'
            elif prob > 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            result = PredictionResult(
                source=source,
                target=target,
                source_type=source_type,
                target_type=target_type,
                probability=prob,
                paths=paths[:3],  # Top 3 paths
                mechanism=mechanism,
                confidence=confidence
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} predicted {source_type} → {target_type} associations")
        return results
    
    def _generate_mechanism_description(
        self,
        source: str,
        target: str,
        paths: List[List[str]]
    ) -> str:
        """Generate natural language description of the predicted mechanism."""
        if not paths:
            return f"No direct pathway found between {source} and {target}."
        
        # Take the shortest path
        path = min(paths, key=len)
        
        # Build description
        descriptions = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            rel = self.edge_types.get((u, v), 'RELATES_TO')
            u_type = self.node_types.get(u, 'entity')
            v_type = self.node_types.get(v, 'entity')
            
            rel_desc = {
                'TARGETS': 'targets',
                'ASSOCIATED_WITH': 'is associated with',
                'INVOLVED_IN': 'is involved in',
                'INTERACTS_WITH': 'interacts with',
                'CAUSES': 'may cause'
            }.get(rel, 'relates to')
            
            descriptions.append(f"{u} ({u_type}) {rel_desc} {v} ({v_type})")
        
        return " → ".join(path) + ". " + "; ".join(descriptions) + "."
    
    # ===== Phase 5: Interpretation =====
    
    def explain_prediction(
        self,
        source: str,
        target: str,
        visualize: bool = True
    ) -> Dict:
        """
        Explain a specific prediction with path analysis.
        
        Args:
            source: Source node
            target: Target node
            visualize: Whether to create visualization
            
        Returns:
            Explanation dictionary
        """
        logger.info(f"Explaining prediction: {source} → {target}")
        
        # Get all paths
        paths = self.visualizer.trace_path(source, target, max_length=5)
        
        explanation = {
            'source': source,
            'target': target,
            'source_type': self.node_types.get(source, 'Unknown'),
            'target_type': self.node_types.get(target, 'Unknown'),
            'num_paths': len(paths),
            'paths': paths,
            'mechanisms': []
        }
        
        # Analyze each path
        for path in paths[:5]:  # Top 5 paths
            mechanism = {
                'path': path,
                'length': len(path),
                'edges': []
            }
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_info = {
                    'source': u,
                    'target': v,
                    'source_type': self.node_types.get(u, 'Unknown'),
                    'target_type': self.node_types.get(v, 'Unknown'),
                    'relationship': self.edge_types.get((u, v), 'Unknown')
                }
                mechanism['edges'].append(edge_info)
            
            explanation['mechanisms'].append(mechanism)
        
        # Visualize shortest path
        if visualize and paths:
            shortest_path = min(paths, key=len)
            self.visualizer.visualize_path(
                shortest_path,
                save_path=str(Path(self.config.output_dir) / f"path_{source}_{target}.png")
            )
        
        return explanation
    
    def visualize_embeddings(
        self,
        method: str = 'tsne',
        save_path: str = None
    ):
        """
        Visualize learned embeddings in 2D.
        
        Compares with force-directed layout to validate
        that GNN learned meaningful structure.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to visualize")
        
        save_path = save_path or str(Path(self.config.output_dir) / "embedding_projection.png")
        
        self.visualizer.visualize_embeddings_2d(
            self.embeddings,
            method=method,
            color_by='node_type',
            save_path=save_path
        )
    
    def compare_visualizations(self, save_path: str = None):
        """Compare force-directed layout with embedding projection."""
        save_path = save_path or str(Path(self.config.output_dir) / "layout_comparison.png")
        
        self.visualizer.compare_layouts(
            embeddings=self.embeddings,
            save_path=save_path
        )
    
    # ===== Full Pipeline =====
    
    def run_full_pipeline(
        self,
        data_source: str = 'csv',
        **kwargs
    ) -> Dict:
        """
        Run the complete integrated pipeline.
        
        Args:
            data_source: 'csv' or 'neo4j'
            **kwargs: Additional arguments for data loading
            
        Returns:
            Complete results dictionary
        """
        logger.info("🚀 Starting Full Cardiotoxicity KG Pipeline...")
        results = {}
        
        # Phase 1: Load data
        logger.info("=" * 50)
        logger.info("PHASE 1: Data Loading")
        if data_source == 'neo4j':
            self.load_graph_from_neo4j()
        else:
            self.load_graph_from_csv(**kwargs)
        results['graph_stats'] = self.visualizer.get_graph_statistics()
        
        # Phase 2: Structural analysis
        logger.info("=" * 50)
        logger.info("PHASE 2: Structural Analysis")
        results['structure'] = self.analyze_structure()
        
        # Phase 3: GNN embedding
        logger.info("=" * 50)
        logger.info("PHASE 3: GNN Embedding")
        results['embedding'] = self.train_embeddings()
        
        # Phase 4: Link prediction
        logger.info("=" * 50)
        logger.info("PHASE 4: Link Prediction")
        results['link_prediction'] = self.train_link_predictor()
        
        # Generate predictions
        predictions = []
        
        # Drug → Disease predictions
        if 'Drug' in set(self.node_types.values()) and 'Disease' in set(self.node_types.values()):
            predictions.extend(self.predict_links('Drug', 'Disease', top_k=10))
        
        # Drug → Gene/Protein predictions
        for target_type in ['Gene', 'Protein']:
            if 'Drug' in set(self.node_types.values()) and target_type in set(self.node_types.values()):
                predictions.extend(self.predict_links('Drug', target_type, top_k=10))
        
        results['predictions'] = [p.to_dict() for p in predictions]
        
        # Phase 5: Visualization comparison
        logger.info("=" * 50)
        logger.info("PHASE 5: Visualization & Interpretation")
        self.compare_visualizations()
        
        logger.info("=" * 50)
        logger.info("✅ Pipeline Complete!")
        logger.info(f"Results saved to: {self.config.output_dir}")
        
        return results


def main():
    """Demo the integrated pipeline."""
    # Create configuration
    config = PipelineConfig(
        raw_data_dir="data/raw",
        output_dir="output",
        embedding_dim=64,
        epochs=100
    )
    
    # Initialize pipeline
    pipeline = CardioKGPipeline(config)
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(data_source='csv')
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        
        print(f"\nGraph Statistics:")
        for key, value in results['graph_stats'].items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
        
        print(f"\nGNN Performance:")
        print(f"  Test AUC: {results['embedding']['test_auc']:.4f}")
        
        print(f"\nLink Prediction Performance:")
        print(f"  AUC: {results['link_prediction']['auc']:.4f}")
        print(f"  AP: {results['link_prediction']['ap']:.4f}")
        
        print(f"\nTop Predictions:")
        for pred in results['predictions'][:5]:
            print(f"  {pred['source']} → {pred['target']}: {pred['probability']:.3f} ({pred['confidence']})")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

