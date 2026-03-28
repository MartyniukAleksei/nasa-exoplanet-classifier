"""
Ensemble model architecture for exoplanet classification and property prediction.
Implements multiple ML models with voting and uncertainty estimation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


class ExoplanetEnsembleClassifier:
    """
    Ensemble classifier for exoplanet classification.
    Combines multiple models for robust predictions with uncertainty estimation.
    """
    
    def __init__(self, use_advanced_models=True):
        self.use_advanced_models = use_advanced_models and XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE
        self.models = {}
        self.ensemble = None
        self.classes_ = None
        self.label_encoder = LabelEncoder()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual classifiers for the ensemble."""
        
        # Random Forest - good for feature interactions
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - sequential error correction
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            random_state=42
        )
        
        # Neural Network - captures complex patterns
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        
        # Logistic Regression - baseline linear model
        self.models['lr'] = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        if self.use_advanced_models:
            # XGBoost - state-of-the-art gradient boosting
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                random_state=42,
                n_jobs=-1
            )
            
            # LightGBM - efficient gradient boosting
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                subsample=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
    
    def fit(self, X, y):
        """Train all models in the ensemble."""
        print("Training ensemble models...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Train individual models
        for name, model in self.models.items():
            print(f"  Training {name.upper()}...")
            model.fit(X, y_encoded)
        
        # Create voting ensemble with soft voting
        estimators = [(name, model) for name, model in self.models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Fit ensemble
        print("  Training ensemble...")
        self.ensemble.fit(X, y_encoded)
        
        print("Ensemble training complete!")
        return self
    
    def predict(self, X):
        """Predict class labels."""
        y_encoded = self.ensemble.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.ensemble.predict_proba(X)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimation.
        Returns predictions, probabilities, and uncertainty metrics.
        """
        # Get predictions from all models
        all_predictions = []
        all_probabilities = []
        
        for name, model in self.models.items():
            pred_encoded = model.predict(X)
            all_predictions.append(pred_encoded)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probabilities.append(proba)
        
        # Ensemble prediction
        ensemble_pred_encoded = self.ensemble.predict(X)
        ensemble_pred = self.label_encoder.inverse_transform(ensemble_pred_encoded)
        ensemble_proba = self.ensemble.predict_proba(X)
        
        # Calculate uncertainty metrics
        # 1. Agreement among models
        all_predictions = np.array(all_predictions)
        agreement = np.mean([np.mean(all_predictions == pred) 
                            for pred in all_predictions])
        
        # 2. Entropy of probability distribution
        entropy = -np.sum(ensemble_proba * np.log(ensemble_proba + 1e-10), axis=1)
        max_entropy = np.log(len(self.classes_))
        normalized_entropy = entropy / max_entropy
        
        # 3. Confidence (max probability)
        confidence = np.max(ensemble_proba, axis=1)
        
        return {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'confidence': confidence,
            'uncertainty': normalized_entropy,
            'model_agreement': agreement
        }
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        print("\n" + "="*60)
        print("ENSEMBLE CLASSIFIER EVALUATION")
        print("="*60)
        
        print(f"\nOverall Accuracy: {accuracy_score(y, predictions):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, predictions))
        
        # Individual model performance
        print("\n" + "-"*60)
        print("Individual Model Performance:")
        print("-"*60)
        for name, model in self.models.items():
            pred = model.predict(X)
            acc = accuracy_score(y, pred)
            print(f"  {name.upper()}: {acc:.4f}")
        
        return accuracy_score(y, predictions)


class ExoplanetPropertyRegressors:
    """
    Regression models for predicting planet physical properties.
    Separate regressors for each property with uncertainty estimation.
    """
    
    def __init__(self, use_advanced_models=True):
        self.use_advanced_models = use_advanced_models and XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE
        self.regressors = {
            'planet_radius': {},
            'planet_temp': {},
            'semi_major_axis': {},
            'impact_parameter': {}
        }
        self._initialize_regressors()
    
    def _initialize_regressors(self):
        """Initialize regression models for each property."""
        
        for property_name in self.regressors.keys():
            # Random Forest Regressor
            self.regressors[property_name]['rf'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting Regressor
            self.regressors[property_name]['gb'] = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                random_state=42
            )
            
            if self.use_advanced_models:
                # XGBoost Regressor
                self.regressors[property_name]['xgb'] = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=-1
                )
    
    def fit(self, X, y_dict):
        """
        Train regressors for all properties.
        
        Args:
            X: Feature matrix
            y_dict: Dictionary with keys 'planet_radius', 'planet_temp', etc.
        """
        print("\nTraining property regressors...")
        
        for property_name in self.regressors.keys():
            if property_name not in y_dict:
                continue
            
            y = y_dict[property_name]
            
            # Convert to numpy array and ensure real values
            y = np.array(y, dtype=np.float64)
            if np.iscomplexobj(y):
                y = np.real(y)
            
            # Remove NaN and inf values
            valid_idx = ~(np.isnan(y) | np.isinf(y))
            if not np.any(valid_idx):
                continue
            
            X_valid = X[valid_idx]
            y_valid = y[valid_idx]
            
            if len(y_valid) < 10:  # Skip if too few samples
                print(f"  Skipping {property_name} (insufficient valid samples)")
                continue
            
            print(f"  Training {property_name} regressors...")
            try:
                for name, model in self.regressors[property_name].items():
                    model.fit(X_valid, y_valid)
            except Exception as e:
                print(f"    Warning: Failed to train {name} for {property_name}: {e}")
                continue
        
        print("Property regressor training complete!")
        return self
    
    def predict(self, X):
        """
        Predict all planet properties.
        Returns dictionary with predictions for each property.
        """
        predictions = {}
        
        for property_name, models in self.regressors.items():
            if len(models) == 0:
                predictions[property_name] = None
                continue
            
            # Average predictions from all models
            property_preds = []
            for model in models.values():
                pred = model.predict(X)
                property_preds.append(pred)
            
            predictions[property_name] = np.mean(property_preds, axis=0)
        
        return predictions
    
    def predict_with_uncertainty(self, X):
        """
        Predict properties with uncertainty estimation.
        """
        predictions = {}
        uncertainties = {}
        
        for property_name, models in self.regressors.items():
            if len(models) == 0:
                predictions[property_name] = None
                uncertainties[property_name] = None
                continue
            
            # Get predictions from all models
            property_preds = []
            for model in models.values():
                pred = model.predict(X)
                property_preds.append(pred)
            
            property_preds = np.array(property_preds)
            
            # Mean prediction
            predictions[property_name] = np.mean(property_preds, axis=0)
            
            # Standard deviation as uncertainty
            uncertainties[property_name] = np.std(property_preds, axis=0)
        
        return predictions, uncertainties


if __name__ == "__main__":
    # Test ensemble models
    from data_preprocessing import load_nasa_exoplanet_data, ExoplanetDataPreprocessor
    from feature_engineering import TransitFeatureEngineering
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess data
    df = load_nasa_exoplanet_data()
    
    feature_cols = ['orbital_period', 'transit_duration', 'transit_depth', 
                    'snr', 'stellar_mass', 'stellar_temp', 'stellar_magnitude']
    
    # Feature engineering
    engineer = TransitFeatureEngineering()
    X = engineer.fit_transform(df[feature_cols])
    y = df['classification']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifier
    classifier = ExoplanetEnsembleClassifier()
    classifier.fit(X_train, y_train)
    classifier.evaluate(X_test, y_test)
