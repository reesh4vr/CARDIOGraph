# Quick Setup Guide

## Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
cd CARDIOGRAPH
./setup.sh
```

## Option 2: Manual Setup

### 1. Create Conda Environment

```bash
conda create -n cardiograph python=3.10
conda activate cardiograph
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- Neo4j URI, username, and password
- OpenAI API key (optional, for agentic pipeline)

### 4. Install Neo4j

Follow the detailed instructions in `docs/setup_instructions.md`:
- Download Neo4j Desktop from https://neo4j.com/download/
- Create a database named `cardiograph`
- Start the database

### 5. Download Data

See `docs/data_sources.md` for download links and instructions.

Place downloaded files in `data/raw/` directory.

## Verify Installation

Test that everything is set up correctly:

```bash
# Test Python packages
python -c "import neo4j; print('Neo4j driver OK')"
python -c "import networkx as nx; print('NetworkX OK')"
python -c "import torch; print('PyTorch OK')"

# Test Neo4j connection (after setting up Neo4j)
python -c "
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)
with driver.session() as session:
    result = session.run('RETURN \"Connection successful!\" as message')
    print(result.single()['message'])
driver.close()
"
```

## Next Steps

1. Preprocess data: `python src/preprocess/preprocess_data.py`
2. Build graph: `python src/graph_construction/build_graph.py`
3. Generate embeddings: `python src/embeddings/run_node2vec.py`
4. Explore notebooks: `jupyter notebook notebooks/`

