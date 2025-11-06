# Research Summary: CardioKG & CARDIOGraph

## Background

### CardioKG (Imperial College, 2025)

CardioKG demonstrated that integrating multi-modal imaging phenotypes with biological databases significantly improves gene-disease predictions and drug repurposing for cardiovascular diseases. Key findings include:

- **Multi-modal Integration**: Combining cardiac magnetic resonance (CMR) imaging features with genomic and drug data creates a more comprehensive knowledge graph
- **Improved Predictions**: The integrated approach outperformed baseline models in predicting gene-disease associations, particularly for:
  - Atrial Fibrillation (AF)
  - Heart Failure (HF)
  - Myocardial Infarction (MI)
- **Drug Repurposing**: The knowledge graph enabled identification of novel drug-disease associations for cardiovascular conditions

### CARDIOGraph Adaptation

CARDIOGraph adapts the CardioKG methodology specifically for **cardiotoxicity risk analysis** by:

1. **Focusing on Adverse Events**: Aggregating data from DrugBank, CTD, DisGeNET, STRING, ChEMBL, and FAERS to identify hidden drug-gene-disease links contributing to adverse cardiac events

2. **Toxicity-Centered Design**: The graph structure emphasizes relationships that may indicate cardiotoxicity risk:
   - Drug → Disease (cardiac adverse events)
   - Drug → Gene → Disease (mechanistic pathways)
   - Evidence from FAERS (real-world adverse event reports)

3. **Practical Applications**:
   - **Drug Safety**: Early identification of potential cardiotoxic drugs
   - **Mechanism Discovery**: Understanding pathways through which drugs cause cardiac adverse events
   - **Biomarker Identification**: Finding genes associated with drug-induced cardiotoxicity
   - **Drug Repurposing**: Identifying safer alternatives or new uses for existing drugs

## Key Data Sources

### DrugBank
- Comprehensive drug information database
- Drug targets, interactions, and mechanisms
- Critical for understanding drug-gene relationships

### DisGeNET
- Gene-disease associations with curated evidence
- Provides scores and evidence levels for associations
- Essential for building gene-disease links in the graph

### CTD (Comparative Toxicogenomics Database)
- Chemical-gene-disease relationships
- Focuses on toxicogenomics data
- Important for toxicity-related associations

### STRING
- Protein-protein interaction networks
- Provides interaction confidence scores
- Enables pathway analysis and network-based predictions

### ChEMBL
- Bioactive molecules and their targets
- Drug-target binding affinities
- Complements DrugBank data

### FAERS (FDA Adverse Event Reporting System)
- Real-world adverse event reports
- Provides evidence for drug-disease associations
- Critical for cardiotoxicity risk assessment

## Methodology

### Graph Construction
1. **Node Types**:
   - Drugs (from DrugBank)
   - Genes (from DisGeNET, STRING)
   - Diseases (from DisGeNET, CTD)
   - Proteins (from STRING)

2. **Relationship Types**:
   - `TARGETS`: Drug → Gene (drug targets)
   - `ASSOCIATED_WITH`: Gene → Disease (gene-disease associations)
   - `RELATES_TO`: Drug → Disease (drug-disease relationships, including cardiotoxicity)
   - `INTERACTS_WITH`: Protein → Protein (protein-protein interactions)

### Embedding Generation
- **Node2Vec**: Efficient random walk-based embeddings
- **DVGAE**: Deep variational graph autoencoder for more complex structural learning

### Link Prediction
- Binary classification task: predict existence of relationships
- Uses concatenated node embeddings as features
- Random Forest, Neural Networks, or GNN-based classifiers

## Expected Outcomes

1. **Improved Cardiotoxicity Prediction**: Better identification of drugs with cardiac adverse event risks
2. **Mechanistic Insights**: Understanding biological pathways leading to cardiotoxicity
3. **Drug Repurposing Opportunities**: Finding safer alternatives or new therapeutic uses
4. **Biomarker Discovery**: Identifying genes associated with drug-induced cardiotoxicity

## Limitations & Future Work

- **Data Quality**: Dependent on quality and completeness of source databases
- **Evidence Integration**: Balancing different evidence types and confidence scores
- **Temporal Dynamics**: Current static graph doesn't capture temporal aspects of adverse events
- **Validation**: Requires experimental or clinical validation of predictions

## References

- CardioKG: Imperial College London (2025) - *CardioKG: Integrating Multi-modal Phenotypes for Cardiovascular Knowledge Graph Construction*
- AISCARF: Related work on AI for cardiovascular research
- Neo4j Graph Data Science: Graph algorithms and embeddings documentation

