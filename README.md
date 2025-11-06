# CARDIOGraph

AI-powered knowledge graph integrating drugs, genes, and cardiovascular diseases to uncover hidden cardiotoxicity patterns.

## Overview

CARDIOGraph is a simplified replication of the Imperial College CardioKG pipeline, focusing on integrating biomedical datasets (DrugBank, DisGeNET, CTD, STRING, ChEMBL, FAERS) into a Neo4j-based knowledge graph for cardiotoxicity risk analysis.

The pipeline enables:
- **Data Integration**: Harmonize multiple biomedical databases into a unified graph structure
- **Graph Construction**: Build a Neo4j knowledge graph with nodes for drugs, genes, diseases, and their relationships
- **Embedding Generation**: Create node embeddings using Node2Vec or DVGAE for downstream tasks
- **Link Prediction**: Classify cardiotoxic links using machine learning models
- **Agentic Reasoning**: Query the knowledge graph using natural language with LLM-powered agents

## Core Objective

Integrate biomedical data → build graph → generate embeddings → classify cardiotoxic links.

## Setup Guide

### Prerequisites

- Python 3.10+
- Neo4j Desktop or Neo4j Community Edition
- Conda (recommended) or pip

### Installation

1. Create and activate conda environment:
```bash
conda create -n cardiograph python=3.10
conda activate cardiograph
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Neo4j Setup

1. Install Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)

2. Create a new database named `cardiograph`

3. Start the database

4. Create a `.env` file in the project root with your Neo4j credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Run Instructions

### 1. Build the Knowledge Graph

Process raw data and construct the Neo4j graph:
```bash
python src/graph_construction/build_graph.py
```

### 2. Generate Node Embeddings

Create embeddings using Node2Vec:
```bash
python src/embeddings/run_node2vec.py
```

Or use DVGAE (see `notebooks/DVGAE_Test.ipynb`)

### 3. Train Link Prediction Models

Train models for gene-disease or drug-disease association:
```bash
python src/models/train_predictor.py
```

### 4. Interactive Agentic Interface

Use the LangChain agent for natural language queries:
```bash
python src/agent/agentic_pipeline.py
```

Or explore in the Jupyter notebook:
```bash
jupyter notebook notebooks/AgenticInterface.ipynb
```

## Project Structure

```
CARDIOGraph/
├── data/
│   ├── raw/                      # CSV/TSV/XML/JSON datasets
│   ├── processed/                # Cleaned harmonized datasets
├── src/
│   ├── preprocess/               # Data cleaning + normalization scripts
│   ├── graph_construction/       # Neo4j ingestion + ontology harmonization
│   ├── embeddings/               # Node2Vec / DVGAE embeddings
│   ├── models/                   # ML models for gene-disease or drug-disease link prediction
│   ├── agent/                    # LangChain-based agentic pipeline for query + reasoning
├── notebooks/
│   ├── EDA.ipynb                 # Dataset exploration
│   ├── KG_Build.ipynb            # End-to-end graph creation demo
│   ├── DVGAE_Test.ipynb          # Graph embedding experimentation
│   ├── AgenticInterface.ipynb    # LLM interaction prototype
├── docs/
│   ├── research_summary.md       # Key summary from CardioKG & AISCARF papers
│   ├── setup_instructions.md     # How to install + connect Neo4j
│   ├── model_notes.md            # Notes on model architectures (Transformer vs GNN)
│   ├── data_sources.md           # List + links to databases used
├── requirements.txt
├── README.md
└── .gitignore
```

## Data Sources

The pipeline integrates the following biomedical databases:

- **DrugBank**: Drug information, targets, and interactions
- **DisGeNET**: Gene-disease associations
- **CTD** (Comparative Toxicogenomics Database): Chemical-gene-disease relationships
- **STRING**: Protein-protein interaction networks
- **ChEMBL**: Bioactive molecules and their targets
- **FAERS**: FDA Adverse Event Reporting System for drug safety

See `docs/data_sources.md` for detailed information and download links.

## Model Architecture

CARDIOGraph evaluates two primary architectures:

1. **Graph Neural Network (DVGAE)**: Learns structure + edge direction for link prediction
2. **Transformer Encoder-Decoder (BioBERT/Graph-BERT)**: For text-driven relation extraction and reasoning

See `docs/model_notes.md` for comparative findings, feasibility, compute needs, and explainability.

## Contributing

This is a research project inspired by the Imperial College CardioKG pipeline. For questions or contributions, please refer to the documentation in the `docs/` folder.

## License

See LICENSE file for details.

