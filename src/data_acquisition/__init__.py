"""Data acquisition module for downloading and preparing datasets."""

from .download_datasets import (
    download_kaggle_ppi,
    download_string_ppi,
    download_disgenet_sample,
    download_reactome_pathways,
    create_gene_mapping
)

__all__ = [
    'download_kaggle_ppi',
    'download_string_ppi', 
    'download_disgenet_sample',
    'download_reactome_pathways',
    'create_gene_mapping'
]

