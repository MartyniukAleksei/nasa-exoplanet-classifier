#!/usr/bin/env python3
"""
Main exoplanet classification system.
Trains models and provides terminal interface for predictions.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_nasa_exoplanet_data, ExoplanetDataPreprocessor
from feature_engineering import TransitFeatureEngineering, calculate_planet_properties
from ensemble_models import ExoplanetEnsembleClassifier, ExoplanetPropertyRegressors


class ExoplanetClassificationSystem:
    """
    Complete exoplanet classification system with training and prediction.
    """
    
    def __init__(self):
        self.preprocessor = ExoplanetDataPreprocessor()
        self.feature_engineer = TransitFeatureEngineering()
        self.classifier = None
        self.regressors = None
        self.feature_cols = [
            'orbital_period', 'transit_duration', 'transit_depth', 
            'snr', 'stellar_mass', 'stellar_temp', 'stellar_magnitude'
        ]
        self.is_trained = False
    
    def train(self, df=None, save_models=True):
        """
        Train the complete classification system.
        """
        print("="*70)
        print("EXOPLANET CLASSIFICATION SYSTEM - TRAINING")
        print("="*70)
        
        # Load data if not provided
        if df is None:
            print("\nLoading NASA exoplanet dataset...")
            df = load_nasa_exoplanet_data()
        
        print(f"Loaded {len(df)} observations")
        print(f"\nClass distribution:")
        print(df['classification'].value_counts())
        
        # Prepare features and labels
        X = df[self.feature_cols].copy()
        y = df['classification'].copy()
        
        # Prepare regression targets (for confirmed exoplanets only)
        y_properties = {
            'planet_radius': df['planet_radius'].values,
            'planet_temp': df['planet_temp'].values,
            'semi_major_axis': df['semi_major_axis'].values,
            'impact_parameter': df['impact_parameter'].values
        }
        
        print("\n" + "-"*70)
        print("STEP 1: Feature Engineering")
        print("-"*70)
        X_engineered = self.feature_engineer.fit_transform(X)
        print(f"Created {len(X_engineered.columns)} features from {len(self.feature_cols)} inputs")
        
        print("\n" + "-"*70)
        print("STEP 2: Train-Test Split")
        print("-"*70)
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Split property targets
        train_idx = X_train.index
        test_idx = X_test.index
        
        y_props_train = {k: v[train_idx] for k, v in y_properties.items()}
        y_props_test = {k: v[test_idx] for k, v in y_properties.items()}
        
        print("\n" + "-"*70)
        print("STEP 3: Training Classification Ensemble")
        print("-"*70)
        self.classifier = ExoplanetEnsembleClassifier(use_advanced_models=True)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        print("\nEvaluating on test set...")
        test_accuracy = self.classifier.evaluate(X_test, y_test)
        
        print("\n" + "-"*70)
        print("STEP 4: Training Property Regressors")
        print("-"*70)
        self.regressors = ExoplanetPropertyRegressors(use_advanced_models=True)
        self.regressors.fit(X_train, y_props_train)
        
        # Evaluate regressors on confirmed exoplanets only
        print("\nEvaluating property predictions...")
        confirmed_mask = y_test == 'confirmed_exoplanet'
        if np.any(confirmed_mask):
            X_test_confirmed = X_test[confirmed_mask]
            props_pred, props_uncert = self.regressors.predict_with_uncertainty(X_test_confirmed)
            
            # Get indices of confirmed exoplanets in the test set
            test_indices = y_test.index[confirmed_mask]
            
            for prop_name in y_properties.keys():
                y_true = y_props_test[prop_name][confirmed_mask]
                y_pred_all = props_pred[prop_name]
                
                # Remove NaN values
                valid = ~(np.isnan(y_true) | np.isnan(y_pred_all))
                if np.any(valid):
                    y_pred = y_pred_all[valid]
                    y_true_valid = y_true[valid]
                    mae = np.mean(np.abs(y_pred - y_true_valid))
                    rmse = np.sqrt(np.mean((y_pred - y_true_valid) ** 2))
                    print(f"  {prop_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        self.is_trained = True
        
        # Save models
        if save_models:
            print("\n" + "-"*70)
            print("STEP 5: Saving Models")
            print("-"*70)
            self.save_models()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Models saved to 'models/' directory")
        print("="*70)
        
        return self
    
    def predict(self, input_features, return_uncertainty=True, show_diagnostic=False):
        """
        Make prediction for new observation.
        
        Args:
            input_features: dict or DataFrame with features
            return_uncertainty: whether to include uncertainty estimates
            show_diagnostic: whether to show diagnostic analysis
        
        Returns:
            dict with classification and properties
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load_models().")
        
        # Show diagnostic if requested
        if show_diagnostic:
            from diagnostic_tool import analyze_observation
            if isinstance(input_features, dict):
                analyze_observation(input_features)
            else:
                analyze_observation(input_features.iloc[0].to_dict())
        
        # Store input for confirmation score calculation
        if isinstance(input_features, dict):
            input_params = input_features
        else:
            input_params = input_features.iloc[0].to_dict()
        
        # Convert input to DataFrame
        if isinstance(input_features, dict):
            input_df = pd.DataFrame([input_features])
        else:
            input_df = input_features.copy()
        
        # Validate input features
        for col in self.feature_cols:
            if col not in input_df.columns:
                raise ValueError(f"Missing required feature: {col}")
        
        # Extract input features in correct order
        X = input_df[self.feature_cols]
        
        # Feature engineering
        X_engineered = self.feature_engineer.transform(X)
        
        # Classification
        if return_uncertainty:
            results = self.classifier.predict_with_uncertainty(X_engineered)
            classification = results['predictions'][0]
            probabilities = results['probabilities'][0]
            confidence = results['confidence'][0]
            uncertainty = results['uncertainty'][0]
        else:
            classification = self.classifier.predict(X_engineered)[0]
            probabilities = self.classifier.predict_proba(X_engineered)[0]
            confidence = np.max(probabilities)
            uncertainty = None
        
        # Property prediction - ONLY for confirmed exoplanets and candidates
        properties = None
        property_uncertainties = None
        
        # Only predict properties for planets, NOT for false positives
        if classification == 'confirmed_exoplanet' or classification == 'planetary_candidate':
            try:
                props_pred, props_uncert = self.regressors.predict_with_uncertainty(X_engineered)
                
                properties = {
                    'planet_radius': float(props_pred['planet_radius'][0]),
                    'planet_temp': float(props_pred['planet_temp'][0]),
                    'semi_major_axis': float(props_pred['semi_major_axis'][0]),
                    'impact_parameter': float(props_pred['impact_parameter'][0])
                }
                
                if return_uncertainty:
                    property_uncertainties = {
                        'planet_radius': float(props_uncert['planet_radius'][0]),
                        'planet_temp': float(props_uncert['planet_temp'][0]),
                        'semi_major_axis': float(props_uncert['semi_major_axis'][0]),
                        'impact_parameter': float(props_uncert['impact_parameter'][0])
                    }
            except Exception as e:
                print(f"Warning: Could not predict properties: {e}")
                properties = None
        
        # Build result dictionary
        result = {
            'classification': classification,
            'confidence': float(confidence)
        }
        
        if return_uncertainty:
            result['uncertainty'] = float(uncertainty)
            result['model_agreement'] = float(results['model_agreement'])
        
        # Add class probabilities
        result['class_probabilities'] = {
            cls: float(prob) 
            for cls, prob in zip(self.classifier.classes_, probabilities)
        }
        
        # Apply confirmation score correction ONLY for borderline cases
        # Don't override clear false positives!
        try:
            from model_improvements import get_corrected_classification
            corrected_result, conf_score = get_corrected_classification(result, input_params)
            
            # Only use correction if:
            # 1. Original was NOT false_positive with high confidence
            # 2. Correction significantly improves confidence
            original_fp_confidence = result['class_probabilities'].get('false_positive', 0)
            
            if (corrected_result['classification'] == 'confirmed_exoplanet' and 
                corrected_result['confidence'] > result['confidence'] + 0.15 and
                original_fp_confidence < 0.6):  # Don't override clear false positives
                original_class = result['classification']
                result = corrected_result
                result['original_classification'] = original_class
                result['correction_applied'] = True
        except:
            pass  # If correction fails, use original result
        
        if properties is not None:
            result['properties'] = properties
            if property_uncertainties is not None:
                result['property_uncertainties'] = property_uncertainties
        
        return result
    
    def save_models(self, directory='models'):
        """Save trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.pkl'))
        joblib.dump(self.feature_engineer, os.path.join(directory, 'feature_engineer.pkl'))
        joblib.dump(self.classifier, os.path.join(directory, 'classifier.pkl'))
        joblib.dump(self.regressors, os.path.join(directory, 'regressors.pkl'))
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory='models'):
        """Load trained models from disk."""
        try:
            self.preprocessor = joblib.load(os.path.join(directory, 'preprocessor.pkl'))
            self.feature_engineer = joblib.load(os.path.join(directory, 'feature_engineer.pkl'))
            self.classifier = joblib.load(os.path.join(directory, 'classifier.pkl'))
            self.regressors = joblib.load(os.path.join(directory, 'regressors.pkl'))
            self.is_trained = True
            print(f"Models loaded from {directory}/")
            return True
        except FileNotFoundError:
            print(f"Models not found in {directory}/")
            return False


def interactive_terminal_interface():
    """
    Interactive terminal interface for exoplanet classification.
    """
    print("\n" + "="*70)
    print("EXOPLANET CLASSIFICATION SYSTEM")
    print("NASA Kepler/K2/TESS Data Analysis")
    print("="*70)
    
    # Initialize system
    system = ExoplanetClassificationSystem()
    
    # Check if models exist
    if not os.path.exists('models'):
        print("\nNo trained models found. Training new models...")
        system.train()
    else:
        print("\nLoading trained models...")
        if not system.load_models():
            print("Failed to load models. Training new models...")
            system.train()
    
    print("\n" + "="*70)
    print("PREDICTION INTERFACE")
    print("="*70)
    print("\nEnter observation parameters for classification.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        print("-" * 70)
        try:
            # Get user input
            print("\nEnter observation parameters:")
            
            orbital_period = input("  Orbital period (days): ").strip()
            if orbital_period.lower() in ['quit', 'exit']:
                break
            orbital_period = float(orbital_period)
            
            transit_duration = float(input("  Transit duration (hours): ").strip())
            transit_depth = float(input("  Transit depth (relative): ").strip())
            snr = float(input("  Signal-to-noise ratio: ").strip())
            stellar_mass = float(input("  Stellar mass (solar masses): ").strip())
            stellar_temp = float(input("  Stellar temperature (K): ").strip())
            stellar_magnitude = float(input("  Stellar magnitude: ").strip())
            
            # Create input dictionary
            input_features = {
                'orbital_period': orbital_period,
                'transit_duration': transit_duration,
                'transit_depth': transit_depth,
                'snr': snr,
                'stellar_mass': stellar_mass,
                'stellar_temp': stellar_temp,
                'stellar_magnitude': stellar_magnitude
            }
            
            # Make prediction with diagnostic
            print("\nAnalyzing observation...")
            result = system.predict(input_features, return_uncertainty=True, show_diagnostic=True)
            
            # Display results
            print("\n" + "="*70)
            print("CLASSIFICATION RESULTS")
            print("="*70)
            
            print(f"\nClassification: {result['classification'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Uncertainty: {result['uncertainty']:.4f}")
            print(f"Model Agreement: {result['model_agreement']:.2%}")
            
            print("\nClass Probabilities:")
            for cls, prob in result['class_probabilities'].items():
                print(f"  {cls}: {prob:.2%}")
            
            if 'properties' in result:
                print("\n" + "-"*70)
                print("PREDICTED PLANET PROPERTIES")
                print("-"*70)
                
                props = result['properties']
                uncerts = result.get('property_uncertainties', {})
                
                print(f"\n  Planet Radius: {props['planet_radius']:.2f} ± {uncerts.get('planet_radius', 0):.2f} Earth radii")
                print(f"  Planet Temperature: {props['planet_temp']:.0f} ± {uncerts.get('planet_temp', 0):.0f} K")
                print(f"  Semi-major Axis: {props['semi_major_axis']:.4f} ± {uncerts.get('semi_major_axis', 0):.4f} AU")
                print(f"  Impact Parameter: {props['impact_parameter']:.3f} ± {uncerts.get('impact_parameter', 0):.3f}")
            
            print("\n" + "="*70)
            
            # Ask for another prediction
            another = input("\nAnalyze another observation? (yes/no): ").strip().lower()
            if another not in ['yes', 'y']:
                break
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except ValueError as e:
            print(f"\nError: Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\n" + "="*70)
    print("Thank you for using the Exoplanet Classification System!")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            # Training mode
            system = ExoplanetClassificationSystem()
            system.train()
        elif sys.argv[1] == 'predict':
            # Single prediction mode from command line arguments
            if len(sys.argv) < 9:
                print("Usage: python exoplanet_classifier.py predict <orbital_period> <transit_duration> <transit_depth> <snr> <stellar_mass> <stellar_temp> <stellar_magnitude>")
                sys.exit(1)
            
            system = ExoplanetClassificationSystem()
            system.load_models()
            
            input_features = {
                'orbital_period': float(sys.argv[2]),
                'transit_duration': float(sys.argv[3]),
                'transit_depth': float(sys.argv[4]),
                'snr': float(sys.argv[5]),
                'stellar_mass': float(sys.argv[6]),
                'stellar_temp': float(sys.argv[7]),
                'stellar_magnitude': float(sys.argv[8])
            }
            
            result = system.predict(input_features)
            
            print("\nClassification:", result['classification'])
            print("Confidence:", f"{result['confidence']:.2%}")
            
            if 'properties' in result:
                print("\nPlanet Properties:")
                for key, value in result['properties'].items():
                    print(f"  {key}: {value:.4f}")
        else:
            print("Unknown command. Use 'train' or 'predict'")
    else:
        # Interactive mode
        interactive_terminal_interface()
