# miRNA-Candidiasis Bioinformatics

A comprehensive Python-based bioinformatics framework for in silico prediction of candidiasis-related microRNAs. This framework provides tools for data processing, feature engineering, machine learning models, and visualization utilities for miRNA-disease association prediction.

## Features

- **Data Processing**: Utilities for processing data from miRDB and CTD databases
- **Feature Engineering**: Advanced similarity metrics including Jaccard index and hypergeometric tests
- **Machine Learning Models**: Multiple classifiers for miRNA-disease association prediction
- **Visualization**: Comprehensive plotting utilities for data exploration and model evaluation

## Project Structure

```
mirna-candidiasis-bioinformatics/
├── src/
│   └── mirna_candidiasis/
│       ├── __init__.py
│       ├── data_processing/
│       │   ├── __init__.py
│       │   ├── mirdb_processor.py      # miRDB database processor
│       │   └── ctd_processor.py        # CTD database processor
│       ├── feature_engineering/
│       │   ├── __init__.py
│       │   ├── jaccard_similarity.py   # Jaccard similarity calculator
│       │   ├── hypergeometric_test.py  # Hypergeometric test utilities
│       │   └── feature_extractor.py    # Feature extraction utilities
│       ├── models/
│       │   ├── __init__.py
│       │   ├── association_predictor.py # ML models for prediction
│       │   └── model_evaluator.py      # Model evaluation utilities
│       └── visualization/
│           ├── __init__.py
│           ├── plot_utils.py           # General plotting utilities
│           ├── feature_plots.py        # Feature visualization
│           └── performance_plots.py    # Performance visualization
├── examples/
│   └── basic_usage.py                  # Example usage script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tiagomiranda24/mirna-candidiasis-bioinformatics.git
cd mirna-candidiasis-bioinformatics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

from mirna_candidiasis.feature_engineering import JaccardSimilarity, FeatureExtractor
from mirna_candidiasis.models import AssociationPredictor

# Define miRNA targets and disease genes
mirna_targets = {
    'hsa-miR-21': {'PTEN', 'PDCD4', 'TIMP3', 'BCL2'},
    'hsa-miR-155': {'SOCS1', 'TP53INP1', 'FOXO3', 'BCL2'}
}

disease_genes = {
    'Candidiasis': {'PTEN', 'BCL2', 'STAT3', 'NFKB1'}
}

# Calculate Jaccard similarity
similarity = JaccardSimilarity.compute(
    mirna_targets['hsa-miR-21'],
    disease_genes['Candidiasis']
)

# Extract features
extractor = FeatureExtractor(population_size=20000)
features = extractor.extract_combined_features(mirna_targets, disease_genes)

# Train a model
predictor = AssociationPredictor(model_type='random_forest')
X, y = predictor.prepare_data(features, target_column='label')
predictor.train(X, y)
```

## Examples

See the `examples/` directory for detailed usage examples:

```bash
cd examples
python basic_usage.py
```

## Modules

### Data Processing

- **MiRDBProcessor**: Process and analyze miRDB database data
- **CTDProcessor**: Process and analyze CTD (Comparative Toxicogenomics Database) data

### Feature Engineering

- **JaccardSimilarity**: Calculate Jaccard similarity coefficients between sets
- **HypergeometricTest**: Perform hypergeometric tests for enrichment analysis
- **FeatureExtractor**: Extract and combine features for machine learning

### Models

- **AssociationPredictor**: Train and use ML models for miRNA-disease association prediction
  - Supports: Random Forest, Gradient Boosting, SVM, Logistic Regression
- **ModelEvaluator**: Evaluate model performance with various metrics

### Visualization

- **PlotUtils**: General-purpose plotting utilities
- **FeaturePlots**: Feature analysis and exploration plots
- **PerformancePlots**: Model performance evaluation plots

## License

See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:
```
Miranda, T. (2024). miRNA-Candidiasis Bioinformatics Framework.
GitHub repository: https://github.com/tiagomiranda24/mirna-candidiasis-bioinformatics
```
