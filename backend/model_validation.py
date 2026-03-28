"""
Model validation and evaluation against known exoplanet catalogs.
Implements comprehensive validation metrics and uncertainty calibration.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_nasa_exoplanet_data
from exoplanet_classifier import ExoplanetClassificationSystem


class ModelValidator:
    """
    Comprehensive validation for exoplanet classification models.
    """
    
    def __init__(self, system):
        self.system = system
        self.validation_results = {}
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the classifier.
        """
        print("\n" + "="*70)
        print("CROSS-VALIDATION")
        print("="*70)
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold
            from ensemble_models import ExoplanetEnsembleClassifier
            fold_classifier = ExoplanetEnsembleClassifier(use_advanced_models=True)
            fold_classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_classifier.predict(X_val)
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_val, y_pred)
            scores.append(score)
            
            print(f"Fold {fold}: Accuracy = {score:.4f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\nCross-validation Results:")
        print(f"  Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")
        
        self.validation_results['cv_scores'] = scores
        self.validation_results['cv_mean'] = mean_score
        self.validation_results['cv_std'] = std_score
        
        return mean_score, std_score
    
    def calibration_analysis(self, X, y):
        """
        Analyze calibration of confidence scores.
        """
        print("\n" + "="*70)
        print("CALIBRATION ANALYSIS")
        print("="*70)
        
        # Get predictions with confidence
        results = self.system.classifier.predict_with_uncertainty(X)
        predictions = results['predictions']
        confidence = results['confidence']
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidence, bins)
        
        calibration_data = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence = np.mean(confidence[mask])
                bin_accuracy = np.mean(predictions[mask] == y[mask])
                bin_count = np.sum(mask)
                
                calibration_data.append({
                    'confidence': bin_confidence,
                    'accuracy': bin_accuracy,
                    'count': bin_count
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        print("\nCalibration Table:")
        print(calibration_df.to_string(index=False))
        
        # Calculate Expected Calibration Error (ECE)
        ece = np.sum(
            np.abs(calibration_df['accuracy'] - calibration_df['confidence']) * 
            calibration_df['count']
        ) / np.sum(calibration_df['count'])
        
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        
        self.validation_results['calibration'] = calibration_df
        self.validation_results['ece'] = ece
        
        return calibration_df, ece
    
    def error_analysis(self, X, y):
        """
        Analyze classification errors in detail.
        """
        print("\n" + "="*70)
        print("ERROR ANALYSIS")
        print("="*70)
        
        # Get predictions
        predictions = self.system.classifier.predict(X)
        
        # Find errors
        errors = predictions != y
        error_indices = np.where(errors)[0]
        
        print(f"\nTotal errors: {np.sum(errors)} / {len(y)} ({np.mean(errors):.2%})")
        
        # Analyze error types
        error_types = {}
        for idx in error_indices:
            true_class = y.iloc[idx]
            pred_class = predictions[idx]
            error_type = f"{true_class} -> {pred_class}"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("\nError Types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
        
        self.validation_results['error_rate'] = np.mean(errors)
        self.validation_results['error_types'] = error_types
        
        return error_types
    
    def uncertainty_quality(self, X, y):
        """
        Evaluate quality of uncertainty estimates.
        """
        print("\n" + "="*70)
        print("UNCERTAINTY QUALITY ASSESSMENT")
        print("="*70)
        
        # Get predictions with uncertainty
        results = self.system.classifier.predict_with_uncertainty(X)
        predictions = results['predictions']
        uncertainty = results['uncertainty']
        confidence = results['confidence']
        
        # Analyze relationship between uncertainty and correctness
        correct = predictions == y
        
        print("\nUncertainty Statistics:")
        print(f"  Correct predictions - Mean uncertainty: {np.mean(uncertainty[correct]):.4f}")
        print(f"  Incorrect predictions - Mean uncertainty: {np.mean(uncertainty[~correct]):.4f}")
        
        print(f"\nConfidence Statistics:")
        print(f"  Correct predictions - Mean confidence: {np.mean(confidence[correct]):.4f}")
        print(f"  Incorrect predictions - Mean confidence: {np.mean(confidence[~correct]):.4f}")
        
        # Uncertainty-accuracy correlation
        correlation = np.corrcoef(uncertainty, ~correct)[0, 1]
        print(f"\nUncertainty-Error Correlation: {correlation:.4f}")
        print("(Higher correlation means uncertainty is more informative)")
        
        self.validation_results['uncertainty_stats'] = {
            'correct_uncertainty': np.mean(uncertainty[correct]),
            'incorrect_uncertainty': np.mean(uncertainty[~correct]),
            'correlation': correlation
        }
        
        return correlation
    
    def property_prediction_validation(self, X, y_properties, classification):
        """
        Validate planet property predictions.
        """
        print("\n" + "="*70)
        print("PROPERTY PREDICTION VALIDATION")
        print("="*70)
        
        # Filter to confirmed exoplanets
        confirmed_mask = classification == 'confirmed_exoplanet'
        X_confirmed = X[confirmed_mask]
        
        if len(X_confirmed) == 0:
            print("No confirmed exoplanets in dataset")
            return
        
        # Get predictions
        props_pred, props_uncert = self.system.regressors.predict_with_uncertainty(X_confirmed)
        
        # Calculate metrics for each property
        property_metrics = {}
        
        for prop_name in ['planet_radius', 'planet_temp', 'semi_major_axis', 'impact_parameter']:
            y_true = y_properties[prop_name][confirmed_mask]
            valid = ~np.isnan(y_true)
            
            if not np.any(valid):
                continue
            
            y_pred = props_pred[prop_name][valid]
            y_true_valid = y_true[valid]
            uncert = props_uncert[prop_name][valid]
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred - y_true_valid))
            rmse = np.sqrt(np.mean((y_pred - y_true_valid) ** 2))
            mape = np.mean(np.abs((y_pred - y_true_valid) / (y_true_valid + 1e-10))) * 100
            
            # R-squared
            ss_res = np.sum((y_true_valid - y_pred) ** 2)
            ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            property_metrics[prop_name] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'mean_uncertainty': np.mean(uncert)
            }
            
            print(f"\n{prop_name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r2:.4f}")
            print(f"  Mean Uncertainty: {np.mean(uncert):.4f}")
        
        self.validation_results['property_metrics'] = property_metrics
        
        return property_metrics
    
    def full_validation_report(self, X, y, y_properties=None):
        """
        Generate comprehensive validation report.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL VALIDATION REPORT")
        print("="*70)
        
        # Cross-validation
        self.cross_validate(X, y, cv=5)
        
        # Calibration analysis
        self.calibration_analysis(X, y)
        
        # Error analysis
        self.error_analysis(X, y)
        
        # Uncertainty quality
        self.uncertainty_quality(X, y)
        
        # Property prediction validation
        if y_properties is not None:
            self.property_prediction_validation(X, y_properties, y)
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"\nCross-validation Accuracy: {self.validation_results['cv_mean']:.4f} ± {self.validation_results['cv_std']:.4f}")
        print(f"Expected Calibration Error: {self.validation_results['ece']:.4f}")
        print(f"Error Rate: {self.validation_results['error_rate']:.2%}")
        print(f"Uncertainty-Error Correlation: {self.validation_results['uncertainty_stats']['correlation']:.4f}")
        
        return self.validation_results


def validate_system():
    """
    Main validation function.
    """
    print("Loading exoplanet data for validation...")
    df = load_nasa_exoplanet_data()
    
    print("Initializing classification system...")
    system = ExoplanetClassificationSystem()
    
    # Check if models are trained
    if not system.load_models():
        print("Training models...")
        system.train(df, save_models=True)
    
    # Prepare data
    feature_cols = [
        'orbital_period', 'transit_duration', 'transit_depth', 
        'snr', 'stellar_mass', 'stellar_temp', 'stellar_magnitude'
    ]
    
    X = df[feature_cols]
    y = df['classification']
    
    y_properties = {
        'planet_radius': df['planet_radius'].values,
        'planet_temp': df['planet_temp'].values,
        'semi_major_axis': df['semi_major_axis'].values,
        'impact_parameter': df['impact_parameter'].values
    }
    
    # Feature engineering
    X_engineered = system.feature_engineer.transform(X)
    
    # Validation
    validator = ModelValidator(system)
    results = validator.full_validation_report(X_engineered, y, y_properties)
    
    return results


if __name__ == "__main__":
    validate_system()
