"""
Train link prediction models for gene-disease and drug-disease associations.

This script trains machine learning models to predict cardiotoxic links
using graph embeddings and node features.
"""

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LinkPredictor:
    """Train models for link prediction in the knowledge graph."""
    
    def __init__(self, embeddings_path='embeddings/node2vec_embeddings.pkl'):
        """Initialize with embeddings."""
        self.embeddings_path = Path(embeddings_path)
        self.embeddings = self.load_embeddings()
        
    def load_embeddings(self):
        """Load node embeddings."""
        if not self.embeddings_path.exists():
            logger.warning(f"Embeddings file not found: {self.embeddings_path}")
            logger.warning("Please run run_node2vec.py first to generate embeddings")
            return None
        
        with open(self.embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"Loaded embeddings for {len(embeddings)} nodes")
        return embeddings
    
    def create_training_data(self, positive_edges, negative_edges=None, negative_ratio=1.0):
        """
        Create training data from positive and negative edges.
        
        Parameters:
        - positive_edges: List of tuples (source, target) for positive examples
        - negative_edges: Optional list of negative examples. If None, will sample negatives.
        - negative_ratio: Ratio of negative to positive examples
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Please generate embeddings first.")
        
        X = []
        y = []
        
        # Process positive edges
        for source, target in positive_edges:
            if source in self.embeddings and target in self.embeddings:
                # Concatenate embeddings
                feature_vector = np.concatenate([
                    self.embeddings[source],
                    self.embeddings[target]
                ])
                X.append(feature_vector)
                y.append(1)
        
        # Generate negative edges if not provided
        if negative_edges is None:
            all_nodes = list(self.embeddings.keys())
            num_negative = int(len(positive_edges) * negative_ratio)
            negative_edges = []
            
            np.random.seed(42)
            for _ in range(num_negative):
                source = np.random.choice(all_nodes)
                target = np.random.choice(all_nodes)
                # Ensure it's not a positive edge
                if (source, target) not in positive_edges:
                    negative_edges.append((source, target))
        
        # Process negative edges
        for source, target in negative_edges:
            if source in self.embeddings and target in self.embeddings:
                feature_vector = np.concatenate([
                    self.embeddings[source],
                    self.embeddings[target]
                ])
                X.append(feature_vector)
                y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created training data: {len(X)} samples ({np.sum(y)} positive, {np.sum(y==0)} negative)")
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train a Random Forest classifier for link prediction.
        
        Returns:
        - model: Trained model
        - X_test, y_test: Test data
        - predictions: Model predictions on test set
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        logger.info("Training Random Forest classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"\nROC-AUC Score: {auc_score:.4f}")
        
        return model, X_test, y_test, y_pred, y_pred_proba
    
    def plot_roc_curve(self, y_test, y_pred_proba, output_path='results/roc_curve.png'):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Link Prediction')
        plt.legend()
        plt.grid(True)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Saved ROC curve to {output_path}")
        plt.close()
    
    def save_model(self, model, output_path='models/link_predictor.pkl'):
        """Save trained model."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved model to {output_path}")


def load_edges_from_neo4j():
    """
    Load positive edges from Neo4j.
    This is a stub - implement based on your specific use case.
    """
    # Example: Load gene-disease associations
    # You would query Neo4j here to get actual edges
    # For now, return empty list as placeholder
    return []


def main():
    """Main execution function."""
    predictor = LinkPredictor()
    
    # Load positive edges (implement based on your data)
    # For gene-disease prediction:
    positive_edges = load_edges_from_neo4j()
    
    if not positive_edges:
        logger.warning("No positive edges loaded. Please implement load_edges_from_neo4j()")
        logger.warning("Example: Load gene-disease or drug-disease associations from Neo4j")
        return
    
    # Create training data
    X, y = predictor.create_training_data(positive_edges, negative_ratio=1.0)
    
    # Train model
    model, X_test, y_test, y_pred, y_pred_proba = predictor.train_model(X, y)
    
    # Plot ROC curve
    predictor.plot_roc_curve(y_test, y_pred_proba)
    
    # Save model
    predictor.save_model(model)
    
    logger.info("Model training completed!")


if __name__ == '__main__':
    main()

