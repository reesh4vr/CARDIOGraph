# CARDIOGraph - Explained for Beginners

## 🎯 What Does This Project Do?

Imagine you're a doctor or researcher trying to answer this question: **"Which drugs might cause heart problems?"**

This is really hard because:
- There are thousands of drugs
- There are thousands of genes that affect the heart
- There are many heart diseases
- The connections between drugs, genes, and diseases are scattered across many different databases

**CARDIOGraph solves this by:**
1. **Collecting** data from 6 major biomedical databases (like DrugBank, DisGeNET, etc.)
2. **Connecting** all this data into a "knowledge graph" (think of it like a giant web showing how everything relates)
3. **Finding patterns** that humans might miss
4. **Answering questions** like "Which drugs are cardiotoxic?" using artificial intelligence

## 📊 What is a Knowledge Graph?

Think of a knowledge graph like a family tree, but for drugs, genes, and diseases:

```
Drug (Aspirin) → Targets → Gene (COX1) → Associated With → Disease (Heart Disease)
```

The graph stores millions of these connections, and we can ask it questions like:
- "What drugs target this gene?"
- "What diseases are linked to this gene?"
- "Show me the pathway from drug X to disease Y"

## 📁 What's in Each Folder?

### `data/`
- **`raw/`**: Place to put downloaded data files (like CSV, XML files from databases)
- **`processed/`**: Cleaned-up versions of that data, ready to use

### `src/` (Source Code)
This is where the "magic" happens - the actual code that does the work:

#### `preprocess/`
- **What it does**: Cleans and organizes messy data files
- **Like**: Taking a messy spreadsheet and making it neat and organized

#### `graph_construction/`
- **What it does**: Builds the knowledge graph in Neo4j (a database)
- **Like**: Taking all the cleaned data and connecting it together like building a web

#### `embeddings/`
- **What it does**: Converts the graph into numbers (embeddings) that computers can understand
- **Like**: Converting words into a code that represents meaning

#### `models/`
- **What it does**: Trains AI models to predict new connections
- **Like**: Teaching a computer to recognize patterns and make predictions

#### `agent/`
- **What it does**: Creates a "smart assistant" that can answer questions in plain English
- **Like**: Having a chatbot that understands your questions about drugs and diseases

### `notebooks/`
Interactive documents where you can explore and experiment. Think of them like digital lab notebooks.

### `docs/`
Documentation - instruction manuals and explanations.

## 📓 What's in Each Notebook?

### 1. `EDA.ipynb` - Exploratory Data Analysis
**What it does**: Helps you understand your data before building the graph

**What you'll see:**
- Charts showing how many drugs, genes, diseases you have
- Statistics about the data
- Visualizations (bar charts, histograms)
- Answers questions like:
  - "How many drugs are in DrugBank?"
  - "What's the distribution of gene-disease association scores?"
  - "Which genes appear most frequently?"

**When to use**: First thing you run after downloading data, to make sure everything looks correct.

---

### 2. `KG_Build.ipynb` - Knowledge Graph Building
**What it does**: Step-by-step guide to building your knowledge graph

**What you'll see:**
- Code that connects to Neo4j database
- How to create nodes (Drugs, Genes, Diseases)
- How to create relationships (Drug → Gene → Disease)
- Queries to check if the graph was built correctly
- Visualizations of sample subgraphs (small portions of your graph)

**Example queries it shows:**
- "Find all drugs with cardiotoxicity associations"
- "Find pathways from drug X to disease Y through genes"
- "Show me graph statistics (how many nodes, how many connections)"

**When to use**: After preprocessing your data, to actually build the graph.

---

### 3. `DVGAE_Test.ipynb` - Deep Learning Experiments
**What it does**: Experiments with advanced AI (Deep Learning) to create better embeddings

**What DVGAE is**: 
- A type of neural network (AI) that learns from the graph structure
- Creates "smart" embeddings that understand relationships better than simple methods
- Can predict missing connections

**What you'll see:**
- How to extract the graph from Neo4j
- How to train a Deep Variational Graph Autoencoder (DVGAE)
- Training progress (loss decreasing over time)
- Performance metrics (how good the predictions are)
- Comparison with simpler methods like Node2Vec

**When to use**: When you want better predictions or more sophisticated AI models.

---

### 4. `AgenticInterface.ipynb` - AI Assistant
**What it does**: Creates a chatbot that can answer questions about your knowledge graph

**What the Agent does:**
- Takes natural language questions (like "Which drugs are cardiotoxic?")
- Searches the Neo4j graph
- Uses AI (like ChatGPT) to give you a natural answer
- Can answer complex questions that require reasoning

**Example questions you can ask:**
- "Which drugs are cardiotoxic?"
- "What are the cardiotoxicity associations for doxorubicin?"
- "What diseases are associated with the gene MYH7?"
- "What genes does aspirin target?"
- "Explain the mechanism by which doxorubicin causes cardiotoxicity"

**What you'll see:**
- Code to initialize the agent
- Example queries and responses
- Interactive mode where you can type questions
- How to use the tools directly (without AI wrapper)

**When to use**: When you want to query your graph using natural language instead of writing code.

## 🤖 What is the Agent?

The **Agent** (`src/agent/agentic_pipeline.py`) is like a smart assistant that combines:

1. **Database Access**: Can query Neo4j to find information
2. **AI Language Model**: Understands your questions and generates answers (uses OpenAI's GPT models)
3. **Tools**: Specialized functions for:
   - Finding cardiotoxic drugs
   - Looking up drug targets
   - Finding gene-disease associations
   - Answering complex questions

**How it works:**
```
You: "Which drugs are cardiotoxic?"
    ↓
Agent: *queries Neo4j graph*
    ↓
Agent: *finds drugs with cardiac disease associations*
    ↓
Agent: *uses AI to format a natural answer*
    ↓
You: "Here are the drugs with cardiotoxicity associations..."
```

**Real example:**
```
Query: "What are the cardiotoxicity associations for doxorubicin?"

Response: "Cardiotoxicity associations for doxorubicin:
- doxorubicin → CAUSES → cardiac arrhythmia (score: 0.85)
- doxorubicin → CAUSES → heart failure (score: 0.92)
- doxorubicin → CAUSES → cardiomyopathy (score: 0.88)"
```

## 🔄 The Complete Workflow

Here's how everything fits together:

1. **Download Data** → Put files in `data/raw/`
2. **Preprocess** → Run `preprocess_data.py` → Clean data goes to `data/processed/`
3. **Explore** → Run `EDA.ipynb` → Understand your data
4. **Build Graph** → Run `build_graph.py` or `KG_Build.ipynb` → Create Neo4j graph
5. **Generate Embeddings** → Run `run_node2vec.py` → Create numerical representations
6. **Train Models** → Run `train_predictor.py` → Train AI to predict new links
7. **Query** → Use `AgenticInterface.ipynb` → Ask questions in natural language

## 🎓 Key Concepts Explained Simply

- **Neo4j**: A special database designed for storing connected data (graphs)
- **Embeddings**: Converting complex data (like graphs) into numbers that computers can work with
- **Link Prediction**: Using AI to predict if two things should be connected (e.g., "Should Drug X be connected to Disease Y?")
- **Node2Vec**: A simple but effective way to create embeddings from graphs
- **DVGAE**: A more sophisticated AI method that learns better representations
- **LangChain**: A tool for building AI agents that can use databases and tools
- **Knowledge Graph**: A way of storing information that emphasizes relationships between things

## 💡 Why This Matters

Before CARDIOGraph:
- Researchers had to manually search through multiple databases
- Hard to see connections between drugs, genes, and diseases
- Easy to miss important patterns

With CARDIOGraph:
- All data in one connected graph
- AI can find hidden patterns
- Ask questions in plain English
- Predict new drug-disease links before they're discovered experimentally

## 🚀 Getting Started

If you're completely new:
1. Start with `README.md` - overview of the project
2. Read `docs/setup_instructions.md` - how to install everything
3. Run `EDA.ipynb` - explore the data
4. Try `KG_Build.ipynb` - build your first graph
5. Experiment with `AgenticInterface.ipynb` - ask questions

Each notebook has comments explaining what each code block does, so you can learn as you go!

