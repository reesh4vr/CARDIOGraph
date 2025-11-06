"""
Preprocessing scripts for cleaning and harmonizing biomedical datasets.

This module handles:
- Parsing raw data files (DrugBank XML, DisGeNET CSV, etc.)
- Normalizing IDs and names across databases
- Filtering for cardiotoxicity-relevant data
- Outputting cleaned CSV files for graph construction
"""

import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess biomedical datasets for graph construction."""
    
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        """Initialize preprocessor with data directories."""
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_drugbank(self, input_file='drugbank_all_full_database.xml'):
        """Parse DrugBank XML and extract drug-gene target relationships."""
        logger.info("Preprocessing DrugBank data...")
        
        input_path = self.raw_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"DrugBank file not found: {input_path}")
            logger.warning("Please download DrugBank data to data/raw/")
            return None
        
        # Parse XML (this is a simplified version - full parsing is more complex)
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        drugs = []
        drug_targets = []
        
        # DrugBank namespace
        ns = {'db': 'http://www.drugbank.ca'}
        
        for drug in root.findall('db:drug', ns):
            drug_id = drug.find('db:drugbank-id[@primary="true"]', ns)
            if drug_id is None:
                continue
            
            drugbank_id = drug_id.text
            name = drug.find('db:name', ns)
            drug_name = name.text if name is not None else ''
            
            # Extract drug type
            drug_type = drug.find('db:groups/db:group', ns)
            drug_type_text = drug_type.text if drug_type is not None else 'unknown'
            
            # Extract targets
            targets = drug.findall('db:targets/db:target', ns)
            for target in targets:
                gene_name = target.find('db:name', ns)
                gene_id_elem = target.find('db:id', ns)
                
                if gene_name is not None and gene_id_elem is not None:
                    gene_name_text = gene_name.text
                    gene_id = gene_id_elem.text
                    
                    # Extract action
                    action = target.find('db:actions/db:action', ns)
                    action_text = action.text if action is not None else 'unknown'
                    
                    drug_targets.append({
                        'drugbank_id': drugbank_id,
                        'drug_name': drug_name,
                        'gene_id': gene_id,
                        'gene_name': gene_name_text,
                        'action': action_text
                    })
            
            drugs.append({
                'drugbank_id': drugbank_id,
                'name': drug_name,
                'type': drug_type_text
            })
        
        # Save processed data
        drugs_df = pd.DataFrame(drugs)
        drugs_df.to_csv(self.processed_dir / 'drugbank_drugs.csv', index=False)
        logger.info(f"Saved {len(drugs_df)} drugs to processed data")
        
        targets_df = pd.DataFrame(drug_targets)
        targets_df.to_csv(self.processed_dir / 'drugbank_targets.csv', index=False)
        logger.info(f"Saved {len(targets_df)} drug-target relationships to processed data")
        
        return drugs_df, targets_df
    
    def preprocess_disgenet(self, input_file='disgenet_gene_disease_associations.csv'):
        """Preprocess DisGeNET gene-disease associations."""
        logger.info("Preprocessing DisGeNET data...")
        
        input_path = self.raw_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"DisGeNET file not found: {input_path}")
            logger.warning("Please download DisGeNET data to data/raw/")
            return None
        
        # Load DisGeNET data
        df = pd.read_csv(input_path, sep='\t', low_memory=False)
        
        # Filter for cardiovascular diseases
        cardiovascular_keywords = [
            'cardiac', 'heart', 'cardiovascular', 'atrial', 'ventricular',
            'myocardial', 'arrhythmia', 'tachycardia', 'bradycardia',
            'cardiomyopathy', 'heart failure', 'coronary', 'ischemic'
        ]
        
        df_filtered = df[
            df['diseaseName'].str.contains('|'.join(cardiovascular_keywords), 
                                          case=False, na=False)
        ]
        
        # Select relevant columns
        processed_df = df_filtered[[
            'geneId', 'geneSymbol', 'diseaseId', 'diseaseName',
            'score', 'evidenceLevel', 'source'
        ]].copy()
        
        processed_df.columns = [
            'gene_id', 'gene_symbol', 'disease_id', 'disease_name',
            'score', 'evidence_level', 'source'
        ]
        
        # Save processed data
        processed_df.to_csv(self.processed_dir / 'disgenet_gene_disease.csv', index=False)
        logger.info(f"Saved {len(processed_df)} gene-disease associations to processed data")
        
        # Extract unique genes
        genes_df = processed_df[['gene_id', 'gene_symbol']].drop_duplicates()
        genes_df.to_csv(self.processed_dir / 'disgenet_genes.csv', index=False)
        
        # Extract unique diseases
        diseases_df = processed_df[['disease_id', 'disease_name']].drop_duplicates()
        diseases_df.to_csv(self.processed_dir / 'disgenet_diseases.csv', index=False)
        
        return processed_df
    
    def preprocess_ctd(self, chemical_disease_file='ctd_chemical_disease_associations.tsv'):
        """Preprocess CTD chemical-disease associations."""
        logger.info("Preprocessing CTD data...")
        
        input_path = self.raw_dir / chemical_disease_file
        
        if not input_path.exists():
            logger.warning(f"CTD file not found: {input_path}")
            return None
        
        # Load CTD data (skip comment lines)
        df = pd.read_csv(input_path, sep='\t', comment='#', low_memory=False)
        
        # Filter for direct evidence and cardiovascular diseases
        df_filtered = df[
            (df['DirectEvidence'] == 'marker/mechanism') |
            (df['DirectEvidence'] == 'therapeutic')
        ]
        
        # Filter for cardiovascular keywords
        cardiovascular_keywords = ['cardiac', 'heart', 'cardiovascular', 'myocardial']
        df_cv = df_filtered[
            df_filtered['DiseaseName'].str.contains('|'.join(cardiovascular_keywords),
                                                    case=False, na=False)
        ]
        
        # Save processed data
        df_cv.to_csv(self.processed_dir / 'ctd_chemical_disease.csv', index=False)
        logger.info(f"Saved {len(df_cv)} CTD chemical-disease associations")
        
        return df_cv
    
    def preprocess_string(self, input_file='string_protein_interactions.tsv'):
        """Preprocess STRING protein-protein interactions."""
        logger.info("Preprocessing STRING data...")
        
        input_path = self.raw_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"STRING file not found: {input_path}")
            return None
        
        # Load STRING data
        df = pd.read_csv(input_path, sep='\t', low_memory=False)
        
        # Filter by combined score threshold
        score_threshold = 400
        df_filtered = df[df['combined_score'] >= score_threshold]
        
        # Save processed data
        df_filtered.to_csv(self.processed_dir / 'string_interactions.csv', index=False)
        logger.info(f"Saved {len(df_filtered)} STRING interactions (score >= {score_threshold})")
        
        return df_filtered
    
    def normalize_ids(self):
        """Normalize IDs across different databases to enable matching."""
        logger.info("Normalizing IDs across databases...")
        
        # This would involve:
        # - Mapping gene symbols to Entrez IDs
        # - Mapping drug names across databases
        # - Mapping disease IDs (UMLS, MeSH, DOID)
        
        # Placeholder for ID normalization logic
        logger.info("ID normalization (to be implemented)")
    
    def run_all(self):
        """Run all preprocessing steps."""
        logger.info("Starting data preprocessing pipeline...")
        
        self.preprocess_drugbank()
        self.preprocess_disgenet()
        self.preprocess_ctd()
        self.preprocess_string()
        self.normalize_ids()
        
        logger.info("Data preprocessing completed!")


def main():
    """Main execution function."""
    preprocessor = DataPreprocessor()
    preprocessor.run_all()


if __name__ == '__main__':
    main()

