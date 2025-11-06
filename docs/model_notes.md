# Model Architecture Notes

## Overview

CARDIOGraph evaluates two primary architectures for embedding and reasoning in the knowledge graph:

1. **Graph Neural Network (DVGAE)** - Structure-aware embeddings
2. **Transformer Encoder-Decoder (BioBERT/Graph-BERT)** - Text-driven reasoning

This document compares their feasibility, compute requirements, and explainability for cardiotoxicity prediction.

## Architecture 1: Graph Neural Network (DVGAE)

### Description

**Deep Variational Graph Autoencoder (DVGAE)** learns node embeddings by encoding graph structure and edge directionality. It's particularly suited for link prediction tasks.

### Advantages

1. **Structure Learning**: Captures local and global graph structure through message passing
2. **Directional Awareness**: Handles directed relationships (e.g., Drug → Gene → Disease)
3. **Link Prediction**: Naturally suited for predicting missing edges
4. **Scalability**: Can handle large graphs efficiently with batching
5. **Feature Integration**: Can incorporate node attributes (e.g., drug properties, gene functions)

### Implementation Details

```python
# Pseudocode structure
class DVGAE(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        self.encoder = GraphEncoder(...)
        self.decoder = EdgeDecoder(...)
    
    def forward(self, edge_index, node_features):
        # Encode graph structure
        z = self.encoder(edge_index, node_features)
        # Decode to predict edges
        edge_pred = self.decoder(z)
        return z, edge_pred
```

### Compute Requirements

- **Training Time**: Moderate (hours for graphs with 10K+ nodes)
- **Memory**: Moderate (depends on batch size and graph density)
- **GPU**: Recommended for graphs > 5K nodes
- **Inference**: Fast (milliseconds per query)

### Explainability

- **Medium**: Can visualize attention weights in GNN layers
- **Node Importance**: Can use gradient-based methods to identify important nodes
- **Path Analysis**: Can trace message passing paths between nodes
- **Limitation**: Less interpretable than rule-based approaches

### Use Cases

- **Link Prediction**: Predicting drug-disease or gene-disease associations
- **Embedding Generation**: Creating dense representations for downstream tasks
- **Anomaly Detection**: Identifying unusual patterns in the graph

## Architecture 2: Transformer Encoder-Decoder (BioBERT/Graph-BERT)

### Description

**BioBERT** or **Graph-BERT** uses transformer architecture to process text descriptions and graph-structured information. It's useful for reasoning about relationships using natural language.

### Advantages

1. **Text Integration**: Can leverage rich textual descriptions from databases
2. **Pre-trained Models**: BioBERT is pre-trained on biomedical literature
3. **Flexible Reasoning**: Can answer complex queries using attention mechanisms
4. **Multi-modal**: Can combine graph structure with text features
5. **Explainability**: Attention weights show which parts of input are important

### Implementation Details

```python
# Pseudocode structure
class GraphTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.graph_embedding = GraphEmbeddingLayer(...)
        self.transformer = nn.Transformer(...)
    
    def forward(self, text_tokens, graph_adjacency):
        # Combine text and graph embeddings
        x = self.token_embedding(text_tokens) + self.graph_embedding(graph_adjacency)
        # Transformer processing
        output = self.transformer(x)
        return output
```

### Compute Requirements

- **Training Time**: High (days for pre-training, hours for fine-tuning)
- **Memory**: High (large transformer models require significant RAM)
- **GPU**: Essential (recommended: 16GB+ VRAM)
- **Inference**: Moderate (seconds per query)

### Explainability

- **High**: Attention visualization shows which tokens/nodes are important
- **Interpretable**: Can generate explanations using attention weights
- **Query Analysis**: Can trace reasoning paths through attention layers
- **Advantage**: More interpretable than pure GNN approaches

### Use Cases

- **Relation Extraction**: Extracting relationships from text descriptions
- **Query Answering**: Natural language queries about the knowledge graph
- **Summarization**: Generating summaries of drug-disease relationships
- **Agentic Reasoning**: Using LLMs to reason about graph structure

## Comparative Analysis

| Aspect | DVGAE (GNN) | Transformer (BioBERT) |
|--------|-------------|----------------------|
| **Structure Learning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Text Integration** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Training Time** | ⭐⭐⭐⭐ | ⭐⭐ |
| **Memory Usage** | ⭐⭐⭐⭐ | ⭐⭐ |
| **Explainability** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Link Prediction** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Query Answering** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Recommended Approach

### Hybrid Strategy

For CARDIOGraph, we recommend a **hybrid approach**:

1. **Graph Construction & Embeddings**: Use **DVGAE** or **Node2Vec** for initial graph embeddings
   - Faster to train
   - Better at capturing graph structure
   - Good baseline for link prediction

2. **Link Prediction**: Use **GNN-based models** (e.g., GraphSAGE, GAT)
   - Train on known edges
   - Predict missing drug-disease or gene-disease links
   - Efficient for large-scale prediction

3. **Agentic Reasoning**: Use **Transformer/LLM** (LangChain + OpenAI/BioBERT)
   - For natural language queries
   - Text-based reasoning about relationships
   - Generating explanations and summaries

### Implementation Phases

**Phase 1: Baseline (Current)**
- Node2Vec embeddings
- Random Forest classifier
- Simple Cypher queries

**Phase 2: Enhanced Embeddings**
- Implement DVGAE
- Compare with Node2Vec
- Evaluate link prediction performance

**Phase 3: Advanced Reasoning**
- Integrate BioBERT or similar transformer
- Build agentic pipeline with LangChain
- Enable natural language querying

## Performance Benchmarks (Expected)

### Link Prediction (Gene-Disease Association)

| Model | ROC-AUC | Precision | Recall | Training Time |
|-------|---------|-----------|--------|---------------|
| Node2Vec + RF | 0.75-0.85 | 0.70-0.80 | 0.65-0.75 | 10-30 min |
| DVGAE | 0.80-0.90 | 0.75-0.85 | 0.70-0.80 | 1-3 hours |
| Graph-BERT | 0.78-0.88 | 0.72-0.82 | 0.68-0.78 | 4-8 hours |

### Query Answering (Agentic Pipeline)

| Query Type | DVGAE Approach | Transformer Approach |
|------------|----------------|---------------------|
| Simple lookup | Fast (ms) | Moderate (s) |
| Complex reasoning | Limited | Excellent |
| Natural language | Not supported | Supported |

## Future Directions

1. **Graph Attention Networks (GAT)**: For explainable attention-based link prediction
2. **Graph Transformer**: Combining GNN and Transformer advantages
3. **Pre-trained Graph Models**: Using models like GraphMAE or GraphGPT
4. **Multi-modal Integration**: Combining graph, text, and imaging features (like CardioKG)

## References

- **DVGAE**: Kipf & Welling, "Variational Graph Auto-Encoders" (2016)
- **Graph-BERT**: Zhang et al., "Graph-BERT: Only Attention is Needed for Learning Graph Representations" (2020)
- **BioBERT**: Lee et al., "BioBERT: a pre-trained biomedical language representation model" (2019)
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric" (2019)

