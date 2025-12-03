"""
Graph Visualization Module for CARDIOGraph Pipeline.

Provides force-directed layouts, community detection, and hub identification
for structural analysis of the knowledge graph.

This visualization phase informs GNN architecture design by:
- Revealing connectivity patterns and hub nodes
- Identifying community structures
- Detecting potential data quality issues
- Determining optimal depth of message-passing layers
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging

# Optional imports for enhanced visualization
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import community as community_louvain
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Visualization tools for knowledge graph structural analysis.
    
    Key capabilities:
    1. Force-directed layouts for network topology
    2. Community detection for cluster identification
    3. Hub node analysis for important entities
    4. Embedding projection for GNN validation
    """
    
    # Color palette for node types - distinctive and accessible
    NODE_COLORS = {
        'Drug': '#FF6B6B',      # Coral red
        'Gene': '#4ECDC4',      # Teal
        'Protein': '#45B7D1',   # Sky blue
        'Disease': '#96CEB4',   # Sage green
        'Pathway': '#FFEAA7',   # Soft yellow
        'Phenotype': '#DDA0DD', # Plum
        'default': '#B8B8B8'    # Gray
    }
    
    # Edge colors for relationship types
    EDGE_COLORS = {
        'TARGETS': '#E74C3C',
        'ASSOCIATED_WITH': '#3498DB',
        'INTERACTS_WITH': '#2ECC71',
        'INVOLVED_IN': '#9B59B6',
        'CAUSES': '#E67E22',
        'TREATS': '#1ABC9C',
        'default': '#95A5A6'
    }
    
    def __init__(self, graph: nx.Graph = None):
        """Initialize visualizer with optional graph."""
        self.graph = graph
        self.layout_cache = {}
        self.community_labels = None
        self.hub_scores = None
        
    def set_graph(self, graph: nx.Graph):
        """Set or update the graph."""
        self.graph = graph
        self.layout_cache = {}
        self.community_labels = None
        self.hub_scores = None
        
    def compute_force_directed_layout(
        self, 
        algorithm: str = 'spring',
        k: float = None,
        iterations: int = 50,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Compute force-directed layout for graph visualization.
        
        Args:
            algorithm: 'spring' (Fruchterman-Reingold), 'kamada_kawai', or 'spectral'
            k: Optimal distance between nodes (auto-computed if None)
            iterations: Number of iterations for spring layout
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping node IDs to 2D positions
        """
        if self.graph is None:
            raise ValueError("No graph set. Use set_graph() first.")
            
        logger.info(f"Computing {algorithm} layout...")
        
        if algorithm == 'spring':
            # Fruchterman-Reingold force-directed algorithm
            pos = nx.spring_layout(
                self.graph, 
                k=k, 
                iterations=iterations,
                seed=seed
            )
        elif algorithm == 'kamada_kawai':
            # Kamada-Kawai path-length cost function
            pos = nx.kamada_kawai_layout(self.graph)
        elif algorithm == 'spectral':
            # Spectral layout using graph Laplacian eigenvectors
            pos = nx.spectral_layout(self.graph)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        self.layout_cache[algorithm] = pos
        logger.info(f"Layout computed for {len(pos)} nodes")
        return pos
    
    def detect_communities(self, method: str = 'louvain') -> Dict[str, int]:
        """
        Detect community structure in the graph.
        
        Args:
            method: 'louvain', 'label_propagation', or 'modularity'
            
        Returns:
            Dictionary mapping node IDs to community labels
        """
        if self.graph is None:
            raise ValueError("No graph set. Use set_graph() first.")
            
        logger.info(f"Detecting communities using {method}...")
        
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        
        if method == 'louvain':
            if HAS_COMMUNITY:
                communities = community_louvain.best_partition(G_undirected)
            else:
                # Fallback to label propagation
                logger.warning("python-louvain not installed, using label propagation")
                communities = self._label_propagation_communities(G_undirected)
        elif method == 'label_propagation':
            communities = self._label_propagation_communities(G_undirected)
        elif method == 'greedy_modularity':
            from networkx.algorithms.community import greedy_modularity_communities
            community_sets = list(greedy_modularity_communities(G_undirected))
            communities = {}
            for i, comm in enumerate(community_sets):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.community_labels = communities
        num_communities = len(set(communities.values()))
        logger.info(f"Found {num_communities} communities")
        return communities
    
    def _label_propagation_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Fallback community detection using label propagation."""
        from networkx.algorithms.community import label_propagation_communities
        communities = {}
        for i, comm in enumerate(label_propagation_communities(G)):
            for node in comm:
                communities[node] = i
        return communities
    
    def identify_hub_nodes(
        self, 
        metric: str = 'degree',
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Identify hub nodes based on centrality metrics.
        
        Args:
            metric: 'degree', 'betweenness', 'pagerank', or 'eigenvector'
            top_k: Number of top hubs to return
            
        Returns:
            List of (node_id, score) tuples sorted by score
        """
        if self.graph is None:
            raise ValueError("No graph set. Use set_graph() first.")
            
        logger.info(f"Computing {metric} centrality...")
        
        if metric == 'degree':
            scores = dict(self.graph.degree())
        elif metric == 'betweenness':
            scores = nx.betweenness_centrality(self.graph)
        elif metric == 'pagerank':
            scores = nx.pagerank(self.graph)
        elif metric == 'eigenvector':
            try:
                scores = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed, using degree")
                scores = dict(self.graph.degree())
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        self.hub_scores = scores
        
        # Sort and return top-k
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def plot_graph(
        self,
        pos: Dict[str, np.ndarray] = None,
        color_by: str = 'node_type',  # 'node_type', 'community', 'centrality'
        node_size_by: str = 'degree',
        highlight_nodes: List[str] = None,
        title: str = "Knowledge Graph Visualization",
        figsize: Tuple[int, int] = (16, 12),
        save_path: str = None,
        show_labels: bool = False,
        edge_alpha: float = 0.3
    ):
        """
        Visualize the knowledge graph with force-directed layout.
        
        Args:
            pos: Node positions (computed if None)
            color_by: How to color nodes
            node_size_by: How to size nodes ('degree', 'uniform', 'centrality')
            highlight_nodes: Nodes to highlight
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
            show_labels: Whether to show node labels
            edge_alpha: Edge transparency
        """
        if self.graph is None:
            raise ValueError("No graph set. Use set_graph() first.")
            
        # Compute layout if not provided
        if pos is None:
            if 'spring' in self.layout_cache:
                pos = self.layout_cache['spring']
            else:
                pos = self.compute_force_directed_layout()
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Compute node colors
        if color_by == 'node_type':
            colors = self._get_node_type_colors()
        elif color_by == 'community':
            if self.community_labels is None:
                self.detect_communities()
            colors = self._get_community_colors()
        elif color_by == 'centrality':
            if self.hub_scores is None:
                self.identify_hub_nodes()
            colors = self._get_centrality_colors()
        else:
            colors = [self.NODE_COLORS['default']] * self.graph.number_of_nodes()
        
        # Compute node sizes
        if node_size_by == 'degree':
            degrees = dict(self.graph.degree())
            max_deg = max(degrees.values()) if degrees else 1
            sizes = [100 + 500 * (degrees.get(n, 1) / max_deg) for n in self.graph.nodes()]
        elif node_size_by == 'centrality':
            if self.hub_scores is None:
                self.identify_hub_nodes()
            max_score = max(self.hub_scores.values()) if self.hub_scores else 1
            sizes = [100 + 500 * (self.hub_scores.get(n, 0) / max_score) for n in self.graph.nodes()]
        else:
            sizes = [200] * self.graph.number_of_nodes()
        
        # Draw edges
        edge_colors = self._get_edge_colors()
        nx.draw_networkx_edges(
            self.graph, pos, 
            alpha=edge_alpha,
            edge_color=edge_colors,
            arrows=True if self.graph.is_directed() else False,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=colors,
            node_size=sizes,
            alpha=0.9,
            ax=ax
        )
        
        # Highlight specific nodes if requested
        if highlight_nodes:
            highlight_pos = {n: pos[n] for n in highlight_nodes if n in pos}
            nx.draw_networkx_nodes(
                self.graph, highlight_pos,
                nodelist=list(highlight_pos.keys()),
                node_color='#FFD700',
                node_size=[s * 1.5 for s in sizes if s],
                alpha=1.0,
                ax=ax
            )
        
        # Draw labels if requested
        if show_labels:
            # Only show labels for high-degree nodes to avoid clutter
            degrees = dict(self.graph.degree())
            threshold = np.percentile(list(degrees.values()), 90)
            labels = {n: n for n, d in degrees.items() if d >= threshold}
            nx.draw_networkx_labels(
                self.graph, pos,
                labels=labels,
                font_size=8,
                font_color='white',
                ax=ax
            )
        
        # Create legend
        self._add_legend(ax, color_by)
        
        ax.set_title(title, fontsize=16, color='white', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
        return fig, ax
    
    def _get_node_type_colors(self) -> List[str]:
        """Get colors based on node types."""
        colors = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data.get('node_type', node_data.get('label', 'default'))
            colors.append(self.NODE_COLORS.get(node_type, self.NODE_COLORS['default']))
        return colors
    
    def _get_community_colors(self) -> List[str]:
        """Get colors based on community membership."""
        # Generate distinct colors for communities
        num_communities = len(set(self.community_labels.values()))
        cmap = plt.cm.get_cmap('tab20', num_communities)
        
        colors = []
        for node in self.graph.nodes():
            comm_id = self.community_labels.get(node, 0)
            colors.append(cmap(comm_id))
        return colors
    
    def _get_centrality_colors(self) -> List[str]:
        """Get colors based on centrality scores (gradient)."""
        scores = [self.hub_scores.get(n, 0) for n in self.graph.nodes()]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            normalized = [0.5] * len(scores)
        else:
            normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        
        # Use a custom colormap (blue to red)
        cmap = plt.cm.get_cmap('plasma')
        return [cmap(n) for n in normalized]
    
    def _get_edge_colors(self) -> List[str]:
        """Get colors based on edge/relationship types."""
        colors = []
        for u, v in self.graph.edges():
            edge_data = self.graph.edges[u, v]
            rel_type = edge_data.get('relationship', edge_data.get('type', 'default'))
            colors.append(self.EDGE_COLORS.get(rel_type, self.EDGE_COLORS['default']))
        return colors
    
    def _add_legend(self, ax, color_by: str):
        """Add legend to the plot."""
        if color_by == 'node_type':
            # Get unique node types in the graph
            node_types = set()
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get('node_type', node_data.get('label', 'Unknown'))
                node_types.add(node_type)
            
            patches = [
                mpatches.Patch(color=self.NODE_COLORS.get(nt, self.NODE_COLORS['default']), 
                              label=nt)
                for nt in sorted(node_types)
            ]
            ax.legend(handles=patches, loc='upper left', facecolor='#16213e', 
                     labelcolor='white', framealpha=0.9)
    
    def plot_degree_distribution(
        self,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: str = None
    ):
        """Plot degree distribution to assess network topology."""
        if self.graph is None:
            raise ValueError("No graph set.")
            
        degrees = [d for n, d in self.graph.degree()]
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        ax.hist(degrees, bins=50, color='#4ECDC4', alpha=0.8, edgecolor='white')
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_xscale('log')
        
        ax.set_xlabel('Degree', fontsize=12, color='white')
        ax.set_ylabel('Frequency', fontsize=12, color='white')
        ax.set_title('Degree Distribution', fontsize=14, color='white')
        ax.tick_params(colors='white')
        
        # Add statistics
        stats_text = f"Nodes: {self.graph.number_of_nodes()}\n"
        stats_text += f"Edges: {self.graph.number_of_edges()}\n"
        stats_text += f"Avg Degree: {np.mean(degrees):.2f}\n"
        stats_text += f"Max Degree: {max(degrees)}"
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               fontsize=10, color='white',
               bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
        
        plt.show()
        return fig
    
    def visualize_embeddings_2d(
        self,
        embeddings: Dict[str, np.ndarray],
        method: str = 'tsne',
        color_by: str = 'node_type',
        figsize: Tuple[int, int] = (12, 10),
        save_path: str = None
    ):
        """
        Project GNN embeddings to 2D for visualization and validation.
        
        This allows comparison with force-directed layouts to confirm
        the GNN has learned biologically meaningful representations.
        
        Args:
            embeddings: Dictionary mapping node IDs to embedding vectors
            method: 'tsne' or 'pca'
            color_by: How to color points
            figsize: Figure size
            save_path: Path to save figure
        """
        if not HAS_SKLEARN:
            logger.error("sklearn required for embedding visualization")
            return
            
        # Filter to nodes in graph
        nodes = [n for n in self.graph.nodes() if n in embeddings]
        X = np.array([embeddings[n] for n in nodes])
        
        logger.info(f"Projecting {len(nodes)} embeddings to 2D using {method}...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_2d = reducer.fit_transform(X)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            X_2d = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Color by node type
        if color_by == 'node_type':
            node_types = [
                self.graph.nodes[n].get('node_type', 'Unknown') 
                for n in nodes
            ]
            unique_types = list(set(node_types))
            
            for nt in unique_types:
                mask = [t == nt for t in node_types]
                points = X_2d[mask]
                color = self.NODE_COLORS.get(nt, self.NODE_COLORS['default'])
                ax.scatter(points[:, 0], points[:, 1], c=color, 
                          label=nt, alpha=0.7, s=50)
            
            ax.legend(facecolor='#16213e', labelcolor='white')
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c='#4ECDC4', alpha=0.7, s=50)
        
        ax.set_title(f'GNN Embeddings ({method.upper()} Projection)', 
                    fontsize=14, color='white')
        ax.tick_params(colors='white')
        ax.set_xlabel('Dimension 1', color='white')
        ax.set_ylabel('Dimension 2', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
        
        plt.show()
        return fig
    
    def compare_layouts(
        self,
        embeddings: Dict[str, np.ndarray] = None,
        figsize: Tuple[int, int] = (20, 8),
        save_path: str = None
    ):
        """
        Compare force-directed layout with GNN embedding projection.
        
        This validates that the GNN has learned meaningful structure
        by comparing the topological layout with learned representations.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')
        
        # 1. Force-directed layout
        pos = self.compute_force_directed_layout()
        ax = axes[0]
        ax.set_facecolor('#1a1a2e')
        colors = self._get_node_type_colors()
        nx.draw_networkx(
            self.graph, pos, ax=ax,
            node_color=colors, node_size=50,
            with_labels=False, alpha=0.7,
            edge_color='#555555', edge_alpha=0.2
        )
        ax.set_title('Force-Directed Layout', fontsize=12, color='white')
        ax.axis('off')
        
        # 2. Community-colored layout
        ax = axes[1]
        ax.set_facecolor('#1a1a2e')
        self.detect_communities()
        comm_colors = self._get_community_colors()
        nx.draw_networkx(
            self.graph, pos, ax=ax,
            node_color=comm_colors, node_size=50,
            with_labels=False, alpha=0.7,
            edge_color='#555555', edge_alpha=0.2
        )
        ax.set_title('Communities Detected', fontsize=12, color='white')
        ax.axis('off')
        
        # 3. Embedding projection (if provided)
        ax = axes[2]
        ax.set_facecolor('#1a1a2e')
        if embeddings and HAS_SKLEARN:
            nodes = [n for n in self.graph.nodes() if n in embeddings]
            X = np.array([embeddings[n] for n in nodes])
            
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(X)
            
            node_types = [
                self.graph.nodes[n].get('node_type', 'Unknown')
                for n in nodes
            ]
            colors = [
                self.NODE_COLORS.get(nt, self.NODE_COLORS['default'])
                for nt in node_types
            ]
            
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7, s=50)
            ax.set_title('GNN Embedding Projection', fontsize=12, color='white')
        else:
            ax.text(0.5, 0.5, 'Provide embeddings\nto visualize', 
                   ha='center', va='center', fontsize=12, color='white')
            ax.set_title('GNN Embeddings (N/A)', fontsize=12, color='white')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
        
        plt.show()
        return fig
    
    def trace_path(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        Trace all simple paths between source and target nodes.
        
        Used for interpreting GNN predictions by identifying
        mechanistic pathways (e.g., drug → gene → pathway → disease).
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        if self.graph is None:
            raise ValueError("No graph set.")
            
        if source not in self.graph or target not in self.graph:
            logger.warning(f"Source or target not in graph: {source}, {target}")
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            logger.info(f"Found {len(paths)} paths from {source} to {target}")
            return paths
        except nx.NetworkXError as e:
            logger.error(f"Path finding error: {e}")
            return []
    
    def visualize_path(
        self,
        path: List[str],
        figsize: Tuple[int, int] = (12, 6),
        save_path: str = None
    ):
        """
        Visualize a specific path through the graph.
        
        Useful for explaining GNN predictions by showing the
        mechanistic chain linking entities.
        """
        if not path or len(path) < 2:
            logger.warning("Path must have at least 2 nodes")
            return
            
        # Create subgraph for the path
        subgraph = self.graph.subgraph(path)
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Linear layout for path
        pos = {node: (i, 0) for i, node in enumerate(path)}
        
        # Draw edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph.edges.get((u, v), {})
            rel_type = edge_data.get('relationship', 'RELATES_TO')
            
            ax.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=2)
            )
            
            # Add edge label
            mid_x = (pos[u][0] + pos[v][0]) / 2
            ax.text(mid_x, 0.15, rel_type, ha='center', fontsize=9, 
                   color='#888888', style='italic')
        
        # Draw nodes
        for node in path:
            node_data = self.graph.nodes[node]
            node_type = node_data.get('node_type', 'Unknown')
            color = self.NODE_COLORS.get(node_type, self.NODE_COLORS['default'])
            
            circle = plt.Circle(pos[node], 0.08, color=color, zorder=5)
            ax.add_patch(circle)
            
            # Node label
            ax.text(pos[node][0], -0.2, node[:20], ha='center', 
                   fontsize=10, color='white')
            ax.text(pos[node][0], -0.35, f"({node_type})", ha='center',
                   fontsize=8, color='#888888')
        
        ax.set_xlim(-0.5, len(path) - 0.5)
        ax.set_ylim(-0.6, 0.4)
        ax.set_aspect('equal')
        ax.axis('off')
        
        title = f"Path: {path[0]} → {path[-1]}"
        ax.set_title(title, fontsize=14, color='white', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor=fig.get_facecolor())
        
        plt.show()
        return fig
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive graph statistics for structural analysis.
        
        Returns:
            Dictionary with graph metrics
        """
        if self.graph is None:
            raise ValueError("No graph set.")
            
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.is_directed() 
                           else nx.is_connected(self.graph),
        }
        
        # Degree statistics
        degrees = [d for n, d in self.graph.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        stats['std_degree'] = np.std(degrees)
        
        # Node type distribution
        node_types = defaultdict(int)
        for node in self.graph.nodes():
            nt = self.graph.nodes[node].get('node_type', 'Unknown')
            node_types[nt] += 1
        stats['node_type_distribution'] = dict(node_types)
        
        # Edge type distribution
        edge_types = defaultdict(int)
        for u, v in self.graph.edges():
            et = self.graph.edges[u, v].get('relationship', 'Unknown')
            edge_types[et] += 1
        stats['edge_type_distribution'] = dict(edge_types)
        
        # Clustering coefficient (for undirected)
        try:
            G_und = self.graph.to_undirected()
            stats['avg_clustering'] = nx.average_clustering(G_und)
        except:
            stats['avg_clustering'] = None
        
        return stats


def main():
    """Demo visualization with sample graph."""
    # Create sample knowledge graph
    G = nx.DiGraph()
    
    # Add nodes with types
    drugs = ['Aspirin', 'Metformin', 'Doxorubicin', 'Lisinopril']
    genes = ['MYH7', 'TNNT2', 'SCN5A', 'KCNQ1', 'PLN']
    diseases = ['Heart Failure', 'Cardiomyopathy', 'Arrhythmia']
    pathways = ['Cardiac Muscle Contraction', 'Ion Transport']
    
    for d in drugs:
        G.add_node(d, node_type='Drug')
    for g in genes:
        G.add_node(g, node_type='Gene')
    for dis in diseases:
        G.add_node(dis, node_type='Disease')
    for p in pathways:
        G.add_node(p, node_type='Pathway')
    
    # Add edges
    G.add_edge('Doxorubicin', 'MYH7', relationship='TARGETS')
    G.add_edge('Doxorubicin', 'Heart Failure', relationship='CAUSES')
    G.add_edge('MYH7', 'Heart Failure', relationship='ASSOCIATED_WITH')
    G.add_edge('MYH7', 'Cardiomyopathy', relationship='ASSOCIATED_WITH')
    G.add_edge('TNNT2', 'Cardiomyopathy', relationship='ASSOCIATED_WITH')
    G.add_edge('SCN5A', 'Arrhythmia', relationship='ASSOCIATED_WITH')
    G.add_edge('KCNQ1', 'Arrhythmia', relationship='ASSOCIATED_WITH')
    G.add_edge('MYH7', 'Cardiac Muscle Contraction', relationship='INVOLVED_IN')
    G.add_edge('TNNT2', 'Cardiac Muscle Contraction', relationship='INVOLVED_IN')
    G.add_edge('SCN5A', 'Ion Transport', relationship='INVOLVED_IN')
    G.add_edge('Aspirin', 'SCN5A', relationship='TARGETS')
    G.add_edge('Lisinopril', 'PLN', relationship='TARGETS')
    G.add_edge('PLN', 'Heart Failure', relationship='ASSOCIATED_WITH')
    
    # Visualize
    viz = GraphVisualizer(G)
    
    # Print statistics
    stats = viz.get_graph_statistics()
    print("\n📊 Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Plot
    viz.plot_graph(
        color_by='node_type',
        title='CARDIOGraph Knowledge Network',
        show_labels=True
    )
    
    # Show hub nodes
    print("\n🔗 Top Hub Nodes:")
    for node, score in viz.identify_hub_nodes(top_k=5):
        print(f"  {node}: {score}")
    
    # Trace a path
    paths = viz.trace_path('Doxorubicin', 'Heart Failure')
    if paths:
        print(f"\n🔍 Paths from Doxorubicin to Heart Failure:")
        for path in paths:
            print(f"  {' → '.join(path)}")
        viz.visualize_path(paths[0])


if __name__ == '__main__':
    main()

