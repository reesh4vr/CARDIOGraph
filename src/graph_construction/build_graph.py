"""
Build Neo4j knowledge graph from processed biomedical datasets.

This script integrates data from DrugBank, DisGeNET, CTD, STRING, ChEMBL, and FAERS
into a unified Neo4j graph structure for cardiotoxicity analysis.
"""

import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Build and manage Neo4j knowledge graph."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")
    
    def create_indexes(self):
        """Create indexes for faster lookups."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.name)",
            "CREATE INDEX IF NOT EXISTS FOR (di:Disease) ON (di.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Protein) ON (p.uniprot_id)"
        ]
        with self.driver.session() as session:
            for index_query in indexes:
                session.run(index_query)
        logger.info("Indexes created")
    
    def create_drug_nodes(self, drugbank_df):
        """Create Drug nodes from DrugBank data."""
        query = """
        UNWIND $drugs AS drug
        MERGE (d:Drug {drugbank_id: drug.id})
        SET d.name = drug.name,
            d.type = drug.type,
            d.description = drug.description
        """
        with self.driver.session() as session:
            session.run(query, drugs=drugbank_df.to_dict('records'))
        logger.info(f"Created {len(drugbank_df)} drug nodes")
    
    def create_gene_nodes(self, disgenet_df):
        """Create Gene nodes from DisGeNET data."""
        query = """
        UNWIND $genes AS gene
        MERGE (g:Gene {gene_id: gene.gene_id})
        SET g.name = gene.gene_symbol,
            g.entrez_id = gene.gene_id
        """
        with self.driver.session() as session:
            session.run(query, genes=disgenet_df.to_dict('records'))
        logger.info(f"Created {len(disgenet_df)} gene nodes")
    
    def create_disease_nodes(self, disease_df):
        """Create Disease nodes."""
        query = """
        UNWIND $diseases AS disease
        MERGE (di:Disease {disease_id: disease.disease_id})
        SET di.name = disease.name,
            di.type = disease.type,
            di.umls_cui = disease.umls_cui
        """
        with self.driver.session() as session:
            session.run(query, diseases=disease_df.to_dict('records'))
        logger.info(f"Created {len(disease_df)} disease nodes")
    
    def create_drug_gene_relationships(self, drug_gene_df):
        """Create TARGETS relationships between drugs and genes."""
        query = """
        UNWIND $relationships AS rel
        MATCH (d:Drug {drugbank_id: rel.drug_id})
        MATCH (g:Gene {gene_id: rel.gene_id})
        MERGE (d)-[r:TARGETS]->(g)
        SET r.action = rel.action,
            r.known_action = rel.known_action
        """
        with self.driver.session() as session:
            session.run(query, relationships=drug_gene_df.to_dict('records'))
        logger.info(f"Created {len(drug_gene_df)} drug-gene relationships")
    
    def create_gene_disease_relationships(self, gene_disease_df):
        """Create ASSOCIATED_WITH relationships between genes and diseases."""
        query = """
        UNWIND $relationships AS rel
        MATCH (g:Gene {gene_id: rel.gene_id})
        MATCH (di:Disease {disease_id: rel.disease_id})
        MERGE (g)-[r:ASSOCIATED_WITH]->(di)
        SET r.score = rel.score,
            r.evidence_level = rel.evidence_level,
            r.source = rel.source
        """
        with self.driver.session() as session:
            session.run(query, relationships=gene_disease_df.to_dict('records'))
        logger.info(f"Created {len(gene_disease_df)} gene-disease relationships")
    
    def create_drug_disease_relationships(self, drug_disease_df):
        """Create CAUSES or TREATS relationships between drugs and diseases."""
        query = """
        UNWIND $relationships AS rel
        MATCH (d:Drug {drugbank_id: rel.drug_id})
        MATCH (di:Disease {disease_id: rel.disease_id})
        MERGE (d)-[r:RELATES_TO]->(di)
        SET r.relationship_type = rel.relationship_type,
            r.evidence_score = rel.evidence_score,
            r.source = rel.source
        """
        with self.driver.session() as session:
            session.run(query, relationships=drug_disease_df.to_dict('records'))
        logger.info(f"Created {len(drug_disease_df)} drug-disease relationships")
    
    def create_protein_interactions(self, string_df):
        """Create INTERACTS_WITH relationships between proteins from STRING."""
        query = """
        UNWIND $interactions AS inter
        MATCH (p1:Protein {uniprot_id: inter.protein1})
        MATCH (p2:Protein {uniprot_id: inter.protein2})
        MERGE (p1)-[r:INTERACTS_WITH]->(p2)
        SET r.combined_score = inter.combined_score,
            r.source = 'STRING'
        """
        with self.driver.session() as session:
            session.run(query, interactions=string_df.to_dict('records'))
        logger.info(f"Created {len(string_df)} protein-protein interactions")
    
    def build_graph(self, processed_data_dir='data/processed'):
        """Main method to build the complete graph."""
        logger.info("Starting graph construction...")
        
        # Clear existing graph
        self.clear_graph()
        
        # Create indexes
        self.create_indexes()
        
        data_dir = Path(processed_data_dir)
        
        # Load processed datasets (stub - replace with actual data loading)
        # Example structure:
        # drugbank_df = pd.read_csv(data_dir / 'drugbank_processed.csv')
        # disgenet_df = pd.read_csv(data_dir / 'disgenet_processed.csv')
        # disease_df = pd.read_csv(data_dir / 'diseases_processed.csv')
        # drug_gene_df = pd.read_csv(data_dir / 'drug_gene_relationships.csv')
        # gene_disease_df = pd.read_csv(data_dir / 'gene_disease_relationships.csv')
        # drug_disease_df = pd.read_csv(data_dir / 'drug_disease_relationships.csv')
        # string_df = pd.read_csv(data_dir / 'string_interactions.csv')
        
        # Create nodes
        # self.create_drug_nodes(drugbank_df)
        # self.create_gene_nodes(disgenet_df)
        # self.create_disease_nodes(disease_df)
        
        # Create relationships
        # self.create_drug_gene_relationships(drug_gene_df)
        # self.create_gene_disease_relationships(gene_disease_df)
        # self.create_drug_disease_relationships(drug_disease_df)
        # self.create_protein_interactions(string_df)
        
        logger.info("Graph construction completed!")
        logger.info("Note: Uncomment data loading sections when processed data is available")


def main():
    """Main execution function."""
    builder = Neo4jGraphBuilder()
    try:
        builder.build_graph()
    finally:
        builder.close()


if __name__ == '__main__':
    main()

