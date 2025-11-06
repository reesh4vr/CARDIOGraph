# Setup Instructions

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large graphs)
- **Disk Space**: At least 10GB free space for Neo4j database and data files

### Required Software

1. **Python & Conda**
   - Install Miniconda or Anaconda from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
   - Verify installation: `conda --version`

2. **Neo4j**
   - **Option A (Recommended)**: Neo4j Desktop
     - Download from [neo4j.com/download](https://neo4j.com/download/)
     - Free for development use
     - Includes GUI for database management
   
   - **Option B**: Neo4j Community Edition
     - Download from [neo4j.com/download](https://neo4j.com/download/)
     - Requires manual configuration

## Installation Steps

### 1. Clone or Download the Repository

```bash
cd /path/to/your/projects
# If using git:
git clone <repository-url>
cd CARDIOGRAPH
```

### 2. Create Conda Environment

```bash
conda create -n cardiograph python=3.10
conda activate cardiograph
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with specific packages:

- **torch-geometric**: May require specific PyTorch version compatibility
  ```bash
  pip install torch torchvision torchaudio
  pip install torch-geometric
  ```

- **node2vec**: Alternative installation
  ```bash
  pip install git+https://github.com/eliorc/node2vec.git
  ```

### 4. Set Up Neo4j

#### Using Neo4j Desktop (Recommended)

1. **Install Neo4j Desktop**
   - Download and install from [neo4j.com](https://neo4j.com/download/)

2. **Create a New Database**
   - Open Neo4j Desktop
   - Click "Add" → "Local DBMS"
   - Name: `cardiograph`
   - Password: Choose a secure password (remember this!)
   - Version: Latest stable (5.x recommended)
   - Click "Create"

3. **Start the Database**
   - Click the "Start" button next to your `cardiograph` database
   - Wait for status to show "Active"
   - Click "Open" to access Neo4j Browser

4. **Verify Connection**
   - In Neo4j Browser, run: `MATCH (n) RETURN count(n)`
   - Should return `0` for an empty database

#### Using Neo4j Community Edition

1. **Download Neo4j**
   ```bash
   # On macOS with Homebrew
   brew install neo4j
   
   # Or download from neo4j.com
   ```

2. **Start Neo4j**
   ```bash
   neo4j start
   ```

3. **Access Neo4j Browser**
   - Open browser: http://localhost:7474
   - Default username: `neo4j`
   - Default password: `neo4j` (change on first login)

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following content (adjust values to match your setup):

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# OpenAI API (for agentic pipeline - optional)
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Replace `your_password_here` with the password you set for Neo4j.

### 6. Verify Installation

Run a quick test to verify everything is set up correctly:

```bash
python -c "from neo4j import GraphDatabase; print('Neo4j driver OK')"
python -c "import networkx as nx; print('NetworkX OK')"
python -c "import torch; print('PyTorch OK')"
python -c "from langchain.agents import initialize_agent; print('LangChain OK')"
```

### 7. Test Neo4j Connection

Create a test script `test_connection.py`:

```python
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    result = session.run("RETURN 'Connection successful!' as message")
    print(result.single()['message'])

driver.close()
```

Run it:
```bash
python test_connection.py
```

Should output: `Connection successful!`

## Downloading Data

### Data Sources

You'll need to download data from the following sources:

1. **DrugBank**: https://go.drugbank.com/releases/latest
   - Requires registration (free for academic use)
   - Download: `drugbank_all_full_database.xml.zip`

2. **DisGeNET**: https://www.disgenet.org/downloads
   - Free registration required
   - Download gene-disease associations CSV

3. **CTD**: http://ctdbase.org/downloads/
   - Free access
   - Download: Chemical-gene interactions, Chemical-disease associations

4. **STRING**: https://string-db.org/cgi/download
   - Free access
   - Download protein-protein interactions for human

5. **ChEMBL**: https://www.ebi.ac.uk/chembl/
   - Free access
   - Use ChEMBL API or download SQLite database

6. **FAERS**: https://fis.fda.gov/content/Exports/aers_faers_download.htm
   - Free access
   - Download quarterly data files

### Data Organization

Place downloaded files in the `data/raw/` directory:

```
data/raw/
├── drugbank_all_full_database.xml
├── disgenet_gene_disease_associations.csv
├── ctd_chemical_gene_interactions.csv
├── ctd_chemical_disease_associations.csv
├── string_protein_interactions.tsv
└── faers_*.csv
```

## Next Steps

1. **Preprocess Data**: Run preprocessing scripts to clean and harmonize data
2. **Build Graph**: Execute `python src/graph_construction/build_graph.py`
3. **Generate Embeddings**: Run `python src/embeddings/run_node2vec.py`
4. **Explore Notebooks**: Start with `notebooks/EDA.ipynb` for data exploration

## Troubleshooting

### Neo4j Connection Issues

**Problem**: `Unable to connect to Neo4j`

**Solutions**:
- Verify Neo4j is running: Check Neo4j Desktop or run `neo4j status`
- Check URI: Should be `bolt://localhost:7687` for local instances
- Verify firewall settings aren't blocking port 7687
- Check username/password in `.env` file

### Memory Issues

**Problem**: `OutOfMemoryError` when building large graphs

**Solutions**:
- Increase Neo4j heap size:
  - Neo4j Desktop: Settings → Database → Memory → Increase heap size
  - Or edit `neo4j.conf`: `dbms.memory.heap.initial_size=2g`
- Process data in batches
- Use Neo4j's `USING PERIODIC COMMIT` for large imports

### Python Package Installation Issues

**Problem**: Package conflicts or installation failures

**Solutions**:
- Use a fresh conda environment
- Install packages one at a time to identify conflicts
- Check Python version compatibility (3.10+)
- For torch-geometric: Ensure PyTorch version matches requirements

### Data Loading Issues

**Problem**: Files not found or parsing errors

**Solutions**:
- Verify file paths in scripts match your directory structure
- Check file formats match expected (CSV, TSV, XML)
- Ensure files are unzipped if needed
- Check file encoding (UTF-8 recommended)

## Getting Help

- Check the `docs/` folder for additional documentation
- Review Neo4j documentation: https://neo4j.com/docs/
- LangChain documentation: https://python.langchain.com/
- Open an issue on the project repository

