"""
Generate Node2Vec embeddings for Neo4j graph nodes.

This script extracts the graph structure from Neo4j, converts it to NetworkX,
and generates node embeddings using Node2Vec.
"""

import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
import pickle
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphEmbeddingGenerator:
    """Generate node embeddings from Neo4j graph."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def extract_graph_to_networkx(self):
        """Extract Neo4j graph and convert to NetworkX."""
        logger.info("Extracting graph from Neo4j...")
        
        G = nx.DiGraph()
        
        # Get all nodes
        with self.driver.session() as session:
            # Get all nodes with their labels
            nodes_query = """
            MATCH (n)
            RETURN labels(n) as labels, 
                   n.name as name,
                   id(n) as neo4j_id,
                   properties(n) as props
            """
            result = session.run(nodes_query)
            
            node_id_map = {}  # Map Neo4j ID to node name
            
            for record in result:
                labels = record['labels']
                name = record['name']
                neo4j_id = record['neo4j_id']
                props = record['props']
                
                if name:
                    node_id_map[neo4j_id] = name
                    node_type = labels[0] if labels else 'Node'
                    G.add_node(name, node_type=node_type, **props)
            
            # Get all relationships
            edges_query = """
            MATCH (a)-[r]->(b)
            RETURN a.name as source, 
                   type(r) as rel_type,
                   b.name as target,
                   properties(r) as props
            """
            result = session.run(edges_query)
            
            for record in result:
                source = record['source']
                target = record['target']
                rel_type = record['rel_type']
                props = record['props'] or {}
                
                if source and target:
                    G.add_edge(source, target, relationship=rel_type, **props)
        
        logger.info(f"Extracted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def generate_node2vec_embeddings(self, G, dimensions=128, walk_length=30, 
                                    num_walks=200, workers=4, p=1, q=1):
        """
        Generate Node2Vec embeddings.
        
        Parameters:
        - dimensions: Embedding dimension (default: 128)
        - walk_length: Length of random walks (default: 30)
        - num_walks: Number of walks per node (default: 200)
        - workers: Number of worker threads (default: 4)
        - p: Return parameter (default: 1)
        - q: In-out parameter (default: 1)
        """
        logger.info("Generating Node2Vec embeddings...")
        
        # Convert to undirected if needed (Node2Vec works on undirected graphs)
        G_undirected = G.to_undirected()
        
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(
            G_undirected,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q
        )
        
        # Train the model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        logger.info(f"Generated embeddings with dimension {dimensions}")
        return model
    
    def save_embeddings(self, model, output_path='embeddings/node2vec_embeddings.pkl'):
        """Save embeddings to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get embeddings as dictionary
        embeddings = {}
        for node in model.wv.index_to_key:
            embeddings[node] = model.wv[node]
        
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Saved embeddings to {output_path}")
        
        # Also save as CSV for easy inspection
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame.from_dict(embeddings, orient='index')
        df.to_csv(csv_path)
        logger.info(f"Saved embeddings CSV to {csv_path}")
    
    def load_embeddings(self, input_path='embeddings/node2vec_embeddings.pkl'):
        """Load embeddings from file."""
        with open(input_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings
    
    def run(self, output_path='embeddings/node2vec_embeddings.pkl', **kwargs):
        """Main method to generate and save embeddings."""
        # Extract graph
        G = self.extract_graph_to_networkx()
        
        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty. Please build the graph first using build_graph.py")
            return
        
        # Generate embeddings
        model = self.generate_node2vec_embeddings(G, **kwargs)
        
        # Save embeddings
        self.save_embeddings(model, output_path)
        
        logger.info("Embedding generation completed!")


def main():
    """Main execution function."""
    generator = GraphEmbeddingGenerator()
    try:
        generator.run(
            dimensions=128,
            walk_length=30,
            num_walks=200,
            workers=4
        )
    finally:
        generator.close()


if __name__ == '__main__':
    main()

