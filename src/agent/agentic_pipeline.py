"""
LangChain-based agentic pipeline for querying Neo4j knowledge graph.

This agent allows natural language queries about cardiotoxicity, drug-gene-disease
relationships, and can reason about the graph structure.
"""

import os
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jQueryTool:
    """Tool for querying Neo4j graph."""
    
    def __init__(self, uri=None, user=None, password=None):
        """Initialize Neo4j connection."""
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
    
    def query_drug_cardiotoxicity(self, drug_name: str) -> str:
        """Query cardiotoxicity associations for a specific drug."""
        query = """
        MATCH (d:Drug {name: $drug_name})-[r:RELATES_TO]->(di:Disease)
        WHERE di.name CONTAINS 'cardiac' OR di.name CONTAINS 'heart' OR di.name CONTAINS 'cardiovascular'
        RETURN d.name as drug, di.name as disease, r.relationship_type as relationship, 
               r.evidence_score as score
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name)
            records = [dict(record) for record in result]
            
            if not records:
                return f"No cardiotoxicity associations found for {drug_name}"
            
            response = f"Cardiotoxicity associations for {drug_name}:\n"
            for record in records:
                response += f"- {record['drug']} → {record['relationship']} → {record['disease']} (score: {record['score']})\n"
            return response
    
    def query_gene_disease_associations(self, gene_name: str) -> str:
        """Query disease associations for a specific gene."""
        query = """
        MATCH (g:Gene {name: $gene_name})-[r:ASSOCIATED_WITH]->(di:Disease)
        RETURN g.name as gene, di.name as disease, r.score as score, r.evidence_level as evidence
        ORDER BY r.score DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, gene_name=gene_name)
            records = [dict(record) for record in result]
            
            if not records:
                return f"No disease associations found for {gene_name}"
            
            response = f"Disease associations for {gene_name}:\n"
            for record in records:
                response += f"- {record['gene']} → {record['disease']} (score: {record['score']}, evidence: {record['evidence']})\n"
            return response
    
    def query_drug_targets(self, drug_name: str) -> str:
        """Query gene targets for a specific drug."""
        query = """
        MATCH (d:Drug {name: $drug_name})-[r:TARGETS]->(g:Gene)
        RETURN d.name as drug, g.name as gene, r.action as action
        LIMIT 20
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name)
            records = [dict(record) for record in result]
            
            if not records:
                return f"No targets found for {drug_name}"
            
            response = f"Targets for {drug_name}:\n"
            for record in records:
                response += f"- {record['drug']} → targets → {record['gene']} (action: {record['action']})\n"
            return response
    
    def query_cardiotoxic_drugs(self) -> str:
        """Query all drugs with cardiotoxicity associations."""
        query = """
        MATCH (d:Drug)-[r:RELATES_TO]->(di:Disease)
        WHERE di.name CONTAINS 'cardiac' OR di.name CONTAINS 'heart' OR di.name CONTAINS 'cardiovascular'
        RETURN DISTINCT d.name as drug, COUNT(r) as association_count
        ORDER BY association_count DESC
        LIMIT 20
        """
        with self.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
            
            if not records:
                return "No cardiotoxic drugs found in the database"
            
            response = "Drugs with cardiotoxicity associations:\n"
            for record in records:
                response += f"- {record['drug']} ({record['association_count']} associations)\n"
            return response
    
    def general_query(self, query_text: str) -> str:
        """Execute a general Cypher query (for advanced users)."""
        try:
            with self.driver.session() as session:
                result = session.run(query_text)
                records = [dict(record) for record in result]
                return str(records[:10])  # Limit to 10 results
        except Exception as e:
            return f"Error executing query: {str(e)}"


class CardioGraphAgent:
    """LangChain agent for querying the cardiotoxicity knowledge graph."""
    
    def __init__(self, openai_api_key=None):
        """Initialize the agent with tools."""
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. Please set it in .env file for LLM functionality.")
        
        self.query_tool = Neo4jQueryTool()
        
        # Initialize LLM
        if self.api_key:
            self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        else:
            self.llm = None
        
        # Create tools
        self.tools = [
            Tool(
                name="QueryDrugCardiotoxicity",
                func=self.query_tool.query_drug_cardiotoxicity,
                description="Query cardiotoxicity associations for a specific drug. Input should be a drug name."
            ),
            Tool(
                name="QueryGeneDiseaseAssociations",
                func=self.query_tool.query_gene_disease_associations,
                description="Query disease associations for a specific gene. Input should be a gene symbol."
            ),
            Tool(
                name="QueryDrugTargets",
                func=self.query_tool.query_drug_targets,
                description="Query gene targets for a specific drug. Input should be a drug name."
            ),
            Tool(
                name="QueryCardiotoxicDrugs",
                func=self.query_tool.query_cardiotoxic_drugs,
                description="Get a list of all drugs with cardiotoxicity associations. No input required."
            )
        ]
        
        # Initialize agent
        if self.llm:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
        else:
            self.agent = None
    
    def query(self, user_input: str) -> str:
        """Process a natural language query."""
        if not self.agent:
            return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
        
        try:
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"
    
    def close(self):
        """Close connections."""
        self.query_tool.close()


def main():
    """Interactive CLI for the agentic pipeline."""
    agent = CardioGraphAgent()
    
    print("CARDIOGraph Agentic Interface")
    print("Ask questions about cardiotoxicity, drugs, genes, and diseases.")
    print("Type 'quit' to exit.\n")
    
    try:
        while True:
            user_input = input("Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                response = agent.query(user_input)
                print(f"\nResponse: {response}\n")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        agent.close()


if __name__ == '__main__':
    main()

