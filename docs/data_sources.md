# Data Sources

This document lists all biomedical databases integrated into CARDIOGraph, along with download links, data formats, and usage notes.

## Primary Data Sources

### 1. DrugBank

**Description**: Comprehensive database containing drug information, targets, interactions, and mechanisms of action.

**Website**: https://go.drugbank.com/

**Download**: 
- URL: https://go.drugbank.com/releases/latest
- Requires: Free registration (academic use)

**Data Format**: 
- XML (full database)
- CSV (specific exports available)

**Key Fields**:
- Drug ID (DrugBank ID)
- Drug name
- Drug type (small molecule, biotech)
- Target genes/proteins
- Drug-drug interactions
- ADMET properties

**Usage in CARDIOGraph**:
- Creates `Drug` nodes
- Establishes `TARGETS` relationships (Drug → Gene)
- Provides drug properties and mechanisms

**Processing Notes**:
- Large XML file requires parsing
- Extract drug-gene target relationships
- Normalize gene names to match other databases

---

### 2. DisGeNET

**Description**: Platform integrating gene-disease associations from multiple sources with curated evidence levels.

**Website**: https://www.disgenet.org/

**Download**:
- URL: https://www.disgenet.org/downloads
- Requires: Free registration

**Data Format**: CSV, TSV

**Key Fields**:
- Gene ID (Entrez Gene ID)
- Gene symbol
- Disease ID (UMLS CUI, DOID)
- Disease name
- Association score (0-1)
- Evidence level (curated, literature, etc.)
- Source database

**Usage in CARDIOGraph**:
- Creates `Gene` and `Disease` nodes
- Establishes `ASSOCIATED_WITH` relationships (Gene → Disease)
- Provides evidence scores for filtering

**Processing Notes**:
- Focus on cardiovascular diseases (filter by disease type)
- Use association scores for edge weights
- Prioritize curated evidence levels

---

### 3. CTD (Comparative Toxicogenomics Database)

**Description**: Database focusing on chemical-gene-disease relationships, particularly useful for toxicology.

**Website**: http://ctdbase.org/

**Download**:
- URL: http://ctdbase.org/downloads/
- Requires: Free access (no registration)

**Data Format**: TSV, CSV

**Key Files**:
- `CTD_chemical_gene_interactions.tsv`: Chemical-gene interactions
- `CTD_chemical_disease_associations.tsv`: Chemical-disease links
- `CTD_gene_disease_associations.tsv`: Gene-disease associations

**Key Fields**:
- Chemical ID (MeSH, CAS, PubChem)
- Gene ID (Entrez Gene ID)
- Disease ID (MeSH, OMIM)
- Interaction type
- Inference score
- Direct/indirect evidence

**Usage in CARDIOGraph**:
- Provides drug-gene-disease triples
- Complements DrugBank and DisGeNET
- Adds toxicology-specific relationships

**Processing Notes**:
- Map chemical IDs to DrugBank IDs (name matching, PubChem bridging)
- Filter for cardiotoxicity-related terms
- Use inference scores for relationship confidence

---

### 4. STRING

**Description**: Database of known and predicted protein-protein interactions.

**Website**: https://string-db.org/

**Download**:
- URL: https://string-db.org/cgi/download
- Requires: Free access

**Data Format**: TSV

**Key Fields**:
- Protein 1 (UniProt ID)
- Protein 2 (UniProt ID)
- Combined score (0-1000)
- Evidence sources (experimental, database, text mining, etc.)

**Usage in CARDIOGraph**:
- Creates `Protein` nodes
- Establishes `INTERACTS_WITH` relationships
- Enables pathway analysis

**Processing Notes**:
- Focus on human proteins (taxonomy ID: 9606)
- Use combined score threshold (e.g., > 400) for filtering
- Map UniProt IDs to gene IDs for integration

---

### 5. ChEMBL

**Description**: Database of bioactive molecules with drug-like properties and their targets.

**Website**: https://www.ebi.ac.uk/chembl/

**Download**:
- URL: https://www.ebi.ac.uk/chembl/downloads
- Requires: Free access

**Data Format**: 
- SQLite database (recommended)
- CSV exports via API

**Key Fields**:
- Molecule ID (ChEMBL ID)
- Molecule name
- Target ID (ChEMBL target ID)
- Binding affinity (IC50, Ki, etc.)
- Assay type

**Usage in CARDIOGraph**:
- Complements DrugBank drug-target data
- Provides binding affinities for relationships
- Adds experimental evidence

**Processing Notes**:
- Use ChEMBL API or SQLite database
- Map ChEMBL targets to gene IDs
- Filter for high-confidence binding data

---

### 6. FAERS (FDA Adverse Event Reporting System)

**Description**: Database of real-world adverse event reports submitted to the FDA.

**Website**: https://fis.fda.gov/content/Exports/aers_faers_download.htm

**Download**:
- URL: https://fis.fda.gov/content/Exports/aers_faers_download.htm
- Requires: Free access

**Data Format**: ASCII, CSV (after processing)

**Key Files** (per quarter):
- `DEMO`: Demographics
- `DRUG`: Drug information
- `REAC`: Reactions (adverse events)
- `OUTC`: Outcomes

**Key Fields**:
- Drug name (normalized)
- Adverse event (MedDRA terms)
- Patient demographics
- Report date
- Outcome

**Usage in CARDIOGraph**:
- Provides real-world evidence for drug-disease relationships
- Establishes `RELATES_TO` relationships with cardiotoxicity evidence
- Adds temporal information (recent reports)

**Processing Notes**:
- Large files require efficient processing
- Normalize drug names to match DrugBank
- Filter for cardiovascular-related adverse events (MedDRA codes)
- Aggregate reports by drug-adverse event pairs

---

## Data Integration Strategy

### 1. Node Harmonization

**Drugs**:
- Primary: DrugBank IDs
- Map: ChEMBL IDs, PubChem IDs (from CTD)
- Normalize: Drug names (case-insensitive, synonym handling)

**Genes**:
- Primary: Entrez Gene ID
- Map: Gene symbols (standardize using HGNC)
- Map: UniProt IDs (from STRING)

**Diseases**:
- Primary: UMLS CUI (from DisGeNET)
- Map: MeSH IDs (from CTD)
- Map: DOID (from DisGeNET)
- Normalize: Disease names (focus on cardiovascular terms)

### 2. Relationship Mapping

| Source | Relationship Type | Confidence Metric |
|--------|------------------|-------------------|
| DrugBank | Drug → Gene (TARGETS) | Known action |
| DisGeNET | Gene → Disease (ASSOCIATED_WITH) | Score (0-1) |
| CTD | Drug → Gene → Disease | Inference score |
| CTD | Drug → Disease | Direct/indirect evidence |
| STRING | Protein → Protein (INTERACTS_WITH) | Combined score |
| ChEMBL | Drug → Gene (TARGETS) | Binding affinity |
| FAERS | Drug → Disease (RELATES_TO) | Report count, frequency |

### 3. Data Quality Filters

- **Minimum Evidence**: Only include relationships with sufficient evidence
- **Score Thresholds**: 
  - DisGeNET: Score > 0.3
  - STRING: Combined score > 400
  - CTD: Direct evidence preferred
- **Cardiovascular Focus**: Filter diseases for cardiac-related terms
- **Temporal Relevance**: Prioritize recent FAERS reports

## Download Instructions

1. **Create data directory**:
   ```bash
   mkdir -p data/raw
   ```

2. **Download each database** to `data/raw/`:
   - Register where required (DrugBank, DisGeNET)
   - Download latest versions
   - Unzip if necessary

3. **Verify downloads**:
   - Check file sizes match expected
   - Verify file formats (CSV, TSV, XML)

## Data Processing Pipeline

See `src/preprocess/` for scripts that:
1. Parse raw data files
2. Extract relevant fields
3. Normalize IDs and names
4. Filter for cardiotoxicity-relevant data
5. Output cleaned CSV files to `data/processed/`

## License Notes

- **DrugBank**: Academic use only (check license)
- **DisGeNET**: Free for research use
- **CTD**: Public domain
- **STRING**: Free for academic use
- **ChEMBL**: Open data (CC BY-SA 3.0)
- **FAERS**: Public domain (FDA data)

Always check current license terms before commercial use.

## Updates and Maintenance

- **Update Frequency**: 
  - DrugBank: Quarterly
  - DisGeNET: Monthly
  - CTD: Monthly
  - STRING: Quarterly
  - ChEMBL: Monthly
  - FAERS: Quarterly

- **Versioning**: Keep track of download dates and versions
- **Data Refresh**: Re-run preprocessing when new data is available

## References

- DrugBank: Wishart et al., "DrugBank 5.0: a major update to the DrugBank database" (2018)
- DisGeNET: Piñero et al., "The DisGeNET knowledge platform for disease genomics" (2020)
- CTD: Davis et al., "The Comparative Toxicogenomics Database" (2021)
- STRING: Szklarczyk et al., "STRING v11: protein-protein association networks" (2019)
- ChEMBL: Gaulton et al., "The ChEMBL database in 2017" (2017)
- FAERS: https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers

