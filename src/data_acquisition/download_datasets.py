"""
Dataset Acquisition for CARDIOGraph Knowledge Graph Pipeline.

Downloads and prepares datasets from multiple sources:
- Kaggle: Protein-protein interactions
- STRING: High-confidence PPI network
- DisGeNET: Gene-disease associations
- Reactome: Pathway data
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
import zipfile
import gzip
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def ensure_directories():
    """Create necessary directories."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directories ready: {RAW_DATA_DIR}")


def download_kaggle_ppi():
    """
    Download protein-protein interactions dataset from Kaggle.
    
    Dataset: alexandervc/protein-protein-interactions
    Contains: Human protein-protein interaction network
    """
    logger.info("📥 Downloading Kaggle PPI dataset...")
    
    try:
        import kagglehub
        path = kagglehub.dataset_download("alexandervc/protein-protein-interactions")
        logger.info(f"Downloaded to: {path}")
        
        # Copy relevant files to our raw data directory
        source_path = Path(path)
        for file in source_path.glob("*.csv"):
            dest = RAW_DATA_DIR / f"kaggle_ppi_{file.name}"
            shutil.copy(file, dest)
            logger.info(f"Copied {file.name} → {dest}")
        
        for file in source_path.glob("*.txt"):
            dest = RAW_DATA_DIR / f"kaggle_ppi_{file.name}"
            shutil.copy(file, dest)
            logger.info(f"Copied {file.name} → {dest}")
            
        return True
        
    except ImportError:
        logger.warning("kagglehub not installed. Install with: pip install kagglehub")
        logger.info("Alternative: Download manually from https://www.kaggle.com/datasets/alexandervc/protein-protein-interactions")
        return False
    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset: {e}")
        return False


def download_string_ppi(species_id="9606", min_score=700):
    """
    Download STRING protein-protein interactions.
    
    Args:
        species_id: NCBI taxonomy ID (9606 = Homo sapiens)
        min_score: Minimum combined score threshold (0-1000)
    """
    logger.info(f"📥 Downloading STRING PPI (species={species_id}, min_score={min_score})...")
    
    # STRING database links file
    url = f"https://stringdb-downloads.org/download/protein.links.v12.0/{species_id}.protein.links.v12.0.txt.gz"
    
    output_gz = RAW_DATA_DIR / f"string_ppi_{species_id}.txt.gz"
    output_txt = RAW_DATA_DIR / f"string_ppi_{species_id}.txt"
    
    try:
        # Download compressed file
        logger.info(f"Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(output_gz, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Decompress
        with gzip.open(output_gz, 'rb') as f_in:
            with open(output_txt, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        output_gz.unlink()
        
        # Filter by score and save
        logger.info("Filtering high-confidence interactions...")
        df = pd.read_csv(output_txt, sep=' ')
        df_filtered = df[df['combined_score'] >= min_score]
        
        output_filtered = RAW_DATA_DIR / f"string_ppi_filtered_{species_id}.csv"
        df_filtered.to_csv(output_filtered, index=False)
        
        logger.info(f"✅ Saved {len(df_filtered)} high-confidence interactions to {output_filtered}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading STRING data: {e}")
        logger.info("Alternative: Download manually from https://string-db.org/cgi/download")
        return False


def download_disgenet_sample():
    """
    Download DisGeNET gene-disease associations.
    
    Note: Full download requires registration at https://www.disgenet.org/
    This function creates a sample structure for testing.
    """
    logger.info("📥 Preparing DisGeNET data structure...")
    
    # Create sample file structure with cardiotoxicity-relevant diseases
    cardio_diseases = {
        "disease_id": [
            "C0018801", "C0018802", "C0004238", "C0007193", 
            "C0027051", "C0042373", "C0011849", "C0085298"
        ],
        "disease_name": [
            "Heart Failure", "Cardiomyopathy, Dilated", "Atrial Fibrillation", 
            "Cardiomyopathy", "Myocardial Infarction", "Vascular Diseases",
            "Diabetes Mellitus", "Cardiomyopathy, Hypertrophic"
        ],
        "disease_class": [
            "Cardiovascular", "Cardiovascular", "Cardiovascular",
            "Cardiovascular", "Cardiovascular", "Cardiovascular",
            "Metabolic", "Cardiovascular"
        ]
    }
    
    df_diseases = pd.DataFrame(cardio_diseases)
    df_diseases.to_csv(RAW_DATA_DIR / "disgenet_diseases_cardio.csv", index=False)
    
    logger.info("""
    ⚠️  DisGeNET requires registration for full dataset download.
    
    To get the full dataset:
    1. Register at https://www.disgenet.org/signup/
    2. Download: curated_gene_disease_associations.tsv
    3. Place in: data/raw/disgenet_gene_disease.tsv
    
    Sample disease list created for testing.
    """)
    
    return True


def download_reactome_pathways():
    """
    Download Reactome pathway data.
    """
    logger.info("📥 Downloading Reactome pathway data...")
    
    # UniProt to Reactome mapping
    url = "https://reactome.org/download/current/UniProt2Reactome.txt"
    output_file = RAW_DATA_DIR / "uniprot_to_reactome.txt"
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        # Parse and filter for human cardiovascular pathways
        df = pd.read_csv(
            output_file, 
            sep='\t', 
            header=None,
            names=['uniprot_id', 'reactome_id', 'url', 'pathway_name', 'evidence', 'species']
        )
        
        # Filter for human
        df_human = df[df['species'] == 'Homo sapiens']
        
        # Filter for cardiovascular-related pathways
        cardio_keywords = [
            'cardiac', 'heart', 'cardio', 'vascular', 'platelet', 
            'coagulation', 'hemostasis', 'muscle contraction',
            'ion channel', 'calcium', 'potassium'
        ]
        
        pattern = '|'.join(cardio_keywords)
        df_cardio = df_human[
            df_human['pathway_name'].str.lower().str.contains(pattern, na=False)
        ]
        
        output_filtered = RAW_DATA_DIR / "reactome_cardio_pathways.csv"
        df_cardio.to_csv(output_filtered, index=False)
        
        logger.info(f"✅ Saved {len(df_cardio)} cardiovascular pathway mappings to {output_filtered}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Reactome data: {e}")
        return False


def create_gene_mapping():
    """
    Create a gene symbol to UniProt ID mapping.
    Uses HGNC data for standardization.
    """
    logger.info("📥 Creating gene mapping file...")
    
    # This would typically come from HGNC or UniProt
    # Creating a sample structure
    gene_mapping = {
        "gene_symbol": [
            "MYH7", "TNNT2", "SCN5A", "KCNQ1", "RYR2", "PLN", "LMNA",
            "TTN", "MYBPC3", "MYL2", "ACTC1", "TPM1", "MYL3", "CASQ2"
        ],
        "uniprot_id": [
            "P12883", "P45379", "Q14524", "P51787", "Q92736", "P26678", "P02545",
            "Q8WZ42", "Q14896", "P10916", "P68032", "P09493", "P08590", "O14958"
        ],
        "entrez_id": [
            "4625", "7139", "6331", "3784", "6262", "5350", "4000",
            "7273", "4607", "4633", "70", "7168", "4634", "845"
        ],
        "gene_name": [
            "Myosin heavy chain 7", "Troponin T2", "Sodium channel protein type 5",
            "Potassium voltage-gated channel subfamily Q member 1", 
            "Ryanodine receptor 2", "Phospholamban", "Prelamin-A/C",
            "Titin", "Myosin-binding protein C", "Myosin regulatory light chain 2",
            "Actin alpha cardiac muscle 1", "Tropomyosin alpha-1 chain",
            "Myosin light chain 3", "Calsequestrin-2"
        ],
        "disease_association": [
            "HCM,DCM", "HCM,DCM", "LQT,Brugada", "LQT", "CPVT,ARVC", 
            "DCM", "DCM,LMNA-cardiomyopathy", "DCM,HCM", "HCM", "HCM",
            "HCM,DCM", "HCM,DCM", "HCM", "CPVT"
        ]
    }
    
    df = pd.DataFrame(gene_mapping)
    output_file = RAW_DATA_DIR / "gene_mapping_cardio.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Created gene mapping with {len(df)} cardiovascular genes")
    return True


def main():
    """Run all dataset downloads."""
    logger.info("🚀 Starting dataset acquisition...")
    ensure_directories()
    
    results = {
        "Kaggle PPI": download_kaggle_ppi(),
        "STRING PPI": download_string_ppi(),
        "DisGeNET": download_disgenet_sample(),
        "Reactome": download_reactome_pathways(),
        "Gene Mapping": create_gene_mapping()
    }
    
    logger.info("\n" + "="*50)
    logger.info("📊 Download Summary:")
    for dataset, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {dataset}")
    
    logger.info("\n📁 Data saved to: " + str(RAW_DATA_DIR))
    logger.info("""
    Next Steps:
    1. Register at DisGeNET for full gene-disease data
    2. Run the preprocessing script to clean and integrate data
    3. Build the knowledge graph with: python src/graph_construction/build_graph.py
    """)


if __name__ == "__main__":
    main()

