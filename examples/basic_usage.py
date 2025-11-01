"""
Example: Basic Usage of miRNA-Candidiasis Bioinformatics Framework

This example demonstrates the basic usage of the framework for miRNA-disease
association prediction, including feature extraction and model training.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
from mirna_candidiasis.feature_engineering import JaccardSimilarity, HypergeometricTest, FeatureExtractor
from mirna_candidiasis.models import AssociationPredictor, ModelEvaluator


def create_sample_data():
    """Create sample data for demonstration."""
    # Sample miRNA-target gene associations
    mirna_targets = {
        'hsa-miR-21': {'PTEN', 'PDCD4', 'TIMP3', 'BCL2'},
        'hsa-miR-155': {'SOCS1', 'TP53INP1', 'FOXO3', 'BCL2'},
        'hsa-miR-146a': {'IRAK1', 'TRAF6', 'NFKB1', 'PDCD4'},
        'hsa-miR-124': {'CDK6', 'STAT3', 'ITGB1', 'PTEN'},
        'hsa-miR-34a': {'SIRT1', 'BCL2', 'NOTCH1', 'TP53INP1'},
    }
    
    # Sample disease-associated genes
    disease_genes = {
        'Candidiasis': {'PTEN', 'BCL2', 'STAT3', 'NFKB1', 'TP53'},
        'Inflammation': {'SOCS1', 'IRAK1', 'TRAF6', 'NFKB1', 'STAT3'},
        'Cancer': {'PTEN', 'BCL2', 'TP53', 'NOTCH1', 'CDK6'},
    }
    
    return mirna_targets, disease_genes


def main():
    print("=" * 70)
    print("miRNA-Candidiasis Bioinformatics Framework - Example Usage")
    print("=" * 70)
    
    # Create sample data
    print("\n1. Creating sample data...")
    mirna_targets, disease_genes = create_sample_data()
    print(f"   - Number of miRNAs: {len(mirna_targets)}")
    print(f"   - Number of diseases: {len(disease_genes)}")
    
    # Calculate Jaccard similarities
    print("\n2. Calculating Jaccard similarities...")
    for mirna, targets in list(mirna_targets.items())[:2]:
        for disease, genes in disease_genes.items():
            sim = JaccardSimilarity.compute(targets, genes)
            print(f"   - {mirna} vs {disease}: {sim:.3f}")
    
    # Perform hypergeometric tests
    print("\n3. Performing hypergeometric tests...")
    population_size = 20000  # Approximate number of human genes
    
    for mirna, targets in list(mirna_targets.items())[:2]:
        for disease, genes in disease_genes.items():
            overlap, pvalue = HypergeometricTest.compute_from_sets(
                targets, genes, population_size
            )
            print(f"   - {mirna} vs {disease}: overlap={overlap}, p-value={pvalue:.4e}")
    
    # Extract combined features
    print("\n4. Extracting combined features...")
    feature_extractor = FeatureExtractor(population_size=population_size)
    features = feature_extractor.extract_combined_features(
        mirna_targets, disease_genes
    )
    print(f"   - Feature matrix shape: {features.shape}")
    print(f"   - Feature columns: {list(features.columns)}")
    
    # Create synthetic labels for demonstration
    print("\n5. Creating synthetic training data...")
    np.random.seed(42)
    features['label'] = np.random.randint(0, 2, size=len(features))
    
    print(f"   - Positive samples: {features['label'].sum()}")
    print(f"   - Negative samples: {len(features) - features['label'].sum()}")
    
    # Train a model
    print("\n6. Training a Random Forest model...")
    predictor = AssociationPredictor(model_type='random_forest', n_estimators=50)
    
    feature_columns = ['Jaccard_Similarity', 'Overlap', 'Fold_Enrichment', 
                      'Negative_Log_Pvalue', 'miRNA_Target_Count', 'Disease_Gene_Count']
    X, y = predictor.prepare_data(features, target_column='label', 
                                  feature_columns=feature_columns)
    
    # Perform cross-validation
    cv_results = predictor.cross_validate(X, y, cv=3, scoring='accuracy')
    print(f"   - Cross-validation accuracy: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # Train on full data
    predictor.train(X, y)
    print("   - Model trained successfully!")
    
    # Get feature importance
    print("\n7. Feature importance:")
    feature_importance = predictor.get_feature_importance_df()
    if feature_importance is not None:
        for _, row in feature_importance.iterrows():
            print(f"   - {row['Feature']}: {row['Importance']:.3f}")
    
    # Make predictions
    print("\n8. Making predictions...")
    predictions = predictor.predict(X)
    probabilities = predictor.predict_proba(X)
    
    # Evaluate model
    print("\n9. Evaluating model performance...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(
        y, predictions, probabilities[:, 1], model_name='RandomForest'
    )
    
    for metric, value in metrics.items():
        print(f"   - {metric}: {value:.3f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
