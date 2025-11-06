#!/bin/bash

# CARDIOGraph Setup Script
# This script helps set up the conda environment and install dependencies

set -e  # Exit on error

echo "🚀 Setting up CARDIOGraph environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'cardiograph'..."
conda create -n cardiograph python=3.10 -y

# Activate environment
echo "🔧 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cardiograph

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your Neo4j credentials!"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate cardiograph"
echo "2. Edit .env file with your Neo4j credentials"
echo "3. Install Neo4j Desktop (see docs/setup_instructions.md)"
echo "4. Download data files to data/raw/ (see docs/data_sources.md)"
echo "5. Run preprocessing: python src/preprocess/preprocess_data.py"
echo ""

