"""
Build Neo4j knowledge graph from processed biomedical datasets.

This script integrates data from DrugBank, DisGeNET, CTD, STRING, ChEMBL, FAERS,
and custom cardiovascular pathway datasets into a unified Neo4j graph structure
for cardiotoxicity analysis.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
import csv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Neo4jGraphBuilder:
    """Build and manage Neo4j knowledge graph."""

    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
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
            "CREATE INDEX IF NOT EXISTS FOR (p:Protein) ON (p.uniprot_id)",
            "CREATE INDEX IF NOT EXISTS FOR (pw:Pathway) ON (pw.name)"
        ]
        with self.driver.session() as session:
            for query in indexes:
                session.run(query)
        logger.info("Indexes created")

    # 🧬 Load pathway data (Protein → Pathway)
    def load_pathway_data(self, csv_path):
        """Load cardiovascular pathway dataset and integrate into Neo4j."""
        logger.info(f"Loading pathway data from {csv_path} ...")

        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return

        with self.driver.session() as session:
            with open(csv_path, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                logger.info(f"Detected CSV headers: {headers}")

                count = 0
                for row in reader:
                    protein = row.get("uniprot_id")
                    pathway_id = row.get("reactome_pathway_id")
                    pathway_name = row.get("pathway_name")

                    if not (protein and pathway_id and pathway_name):
                        continue

                    session.run("""
                        MERGE (p:Protein {uniprot_id: $protein})
                        MERGE (pw:Pathway {id: $pathway_id})
                        SET pw.name = $pathway_name
                        MERGE (p)-[:INVOLVED_IN]->(pw)
                    """, {
                        "protein": protein.strip(),
                        "pathway_id": pathway_id.strip(),
                        "pathway_name": pathway_name.strip()
                    })
                    count += 1

        logger.info(f"✅ Loaded {count} protein–pathway relationships successfully")

    # 💊 Load drug–protein target data (Drug → Protein)
    def load_drug_protein_data(self, csv_path):
        """Load drug–protein target data and integrate into Neo4j."""
        logger.info(f"Loading drug–protein data from {csv_path} ...")

        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return

        with self.driver.session() as session:
            with open(csv_path, "r", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                logger.info(f"Detected CSV headers: {headers}")

                count = 0
                for row in reader:
                    drug_id = row.get("drugbank_id")
                    drug_name = row.get("drug_name")
                    drug_type = row.get("drug_type")
                    uniprot_id = row.get("uniprot_id")
                    uniprot_name = row.get("uniprot_name")
                    target_class = row.get("target_class")

                    if not (drug_id and uniprot_id):
                        continue

                    session.run("""
                        MERGE (d:Drug {drugbank_id: $drug_id})
                        SET d.name = $drug_name,
                            d.type = $drug_type
                        MERGE (p:Protein {uniprot_id: $uniprot_id})
                        SET p.name = $uniprot_name,
                            p.class = $target_class
                        MERGE (d)-[:TARGETS]->(p)
                    """, {
                        "drug_id": drug_id.strip(),
                        "drug_name": (drug_name or "").strip(),
                        "drug_type": (drug_type or "").strip(),
                        "uniprot_id": uniprot_id.strip(),
                        "uniprot_name": (uniprot_name or "").strip(),
                        "target_class": (target_class or "").strip()
                    })
                    count += 1

        logger.info(f"✅ Loaded {count} drug–protein relationships successfully")

    # ⚙️ Build the graph end-to-end
    def build_graph(self):
        """Main method to build the complete graph."""
        logger.info("🚀 Starting graph construction...")
        self.clear_graph()
        self.create_indexes()

        # Load datasets
        self.load_pathway_data("data/raw/Pathway_Cardiovascular_Filtered.csv")
        self.load_drug_protein_data("data/raw/clean_drug_protein_targets.csv")

        logger.info("✅ Graph construction completed!")
        logger.info("Data integrated: Drug → Protein → Pathway relationships are live.")


def main():
    """Main execution function."""
    builder = Neo4jGraphBuilder()
    try:
        builder.build_graph()
    finally:
        builder.close()


if __name__ == "__main__":
    main()
