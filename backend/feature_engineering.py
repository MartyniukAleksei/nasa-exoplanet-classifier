"""
Feature engineering module for exoplanet classification.
Implements physics-based features from transit photometry.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


class TransitFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Feature engineering based on transit photometry and stellar characteristics.
    Inspired by techniques from Osborn et al. (2022) and NASA's EMAC.
    """
    
    def __init__(self):
        self.feature_names_ = None
    
    def calculate_transit_features(self, df):
        """
        Calculate physics-based features from transit parameters.
        """
        features = df.copy()
        
        # Transit duration to period ratio (T/P)
        # Shorter relative durations suggest smaller planets
        features['duration_period_ratio'] = (
            features['transit_duration'] / 24.0  # Convert hours to days
        ) / features['orbital_period']
        
        # Estimated planet-star radius ratio from transit depth
        # depth ≈ (Rp/Rs)^2
        features['radius_ratio'] = np.sqrt(features['transit_depth'])
        
        # Impact parameter estimate from duration and depth
        # Grazing transits have higher impact parameters
        features['estimated_impact'] = 1 - (
            features['duration_period_ratio'] / 
            (2 * np.sqrt(features['transit_depth']))
        )
        features['estimated_impact'] = features['estimated_impact'].clip(0, 1)
        
        # Transit shape parameter (combination of depth and duration)
        features['transit_shape'] = (
            features['transit_depth'] * 
            features['transit_duration']
        )
        
        # Signal strength metric
        features['signal_strength'] = (
            features['snr'] * 
            np.sqrt(features['transit_depth'])
        )
        
        return features
    
    def calculate_stellar_features(self, df):
        """
        Calculate stellar characteristics relevant for planet detection.
        """
        features = df.copy()
        
        # Stellar density proxy (mass/radius relationship)
        # Higher mass stars are generally larger
        features['stellar_density_proxy'] = (
            features['stellar_mass'] / 
            (features['stellar_temp'] / 5778) ** 4
        )
        
        # Stellar luminosity estimate (mass-luminosity relation)
        # L ∝ M^3.5 for main sequence stars
        features['stellar_luminosity'] = features['stellar_mass'] ** 3.5
        
        # Brightness contrast (affects detection probability)
        features['brightness_metric'] = (
            1.0 / (10 ** (features['stellar_magnitude'] / 2.5))
        )
        
        # Stellar radius estimate from mass and temperature
        # Using Stefan-Boltzmann relation
        features['stellar_radius_estimate'] = (
            features['stellar_mass'] ** 0.8 * 
            (features['stellar_temp'] / 5778) ** 0.5
        )
        
        return features
    
    def calculate_orbital_features(self, df):
        """
        Calculate orbital characteristics from observed parameters.
        """
        features = df.copy()
        
        # Semi-major axis estimate using Kepler's Third Law
        # a^3 / P^2 = G*M / (4*pi^2)
        # Simplified: a ∝ P^(2/3) * M^(1/3)
        features['semimajor_axis_estimate'] = (
            (features['orbital_period'] / 365.25) ** (2/3) * 
            features['stellar_mass'] ** (1/3)
        )
        
        # Orbital velocity estimate
        # v ∝ 1/sqrt(a)
        features['orbital_velocity_proxy'] = (
            1.0 / np.sqrt(features['semimajor_axis_estimate'])
        )
        
        # Insolation flux (energy received from star)
        # F ∝ L / a^2
        features['insolation_flux'] = (
            features['stellar_luminosity'] / 
            features['semimajor_axis_estimate'] ** 2
        )
        
        # Equilibrium temperature estimate
        # Teq ∝ Tstar * sqrt(Rstar / a)
        features['equilibrium_temp_estimate'] = (
            features['stellar_temp'] * 
            np.sqrt(features['stellar_radius_estimate'] / 
                   (2 * features['semimajor_axis_estimate']))
        )
        
        return features
    
    def calculate_detection_features(self, df):
        """
        Calculate features related to detection reliability and quality.
        """
        features = df.copy()
        
        # Multiple Event Statistic (MES) - detection significance
        # MES combines SNR with number of transits
        assumed_num_transits = 10  # Typical for Kepler mission
        features['mes_proxy'] = (
            features['snr'] * np.sqrt(assumed_num_transits)
        )
        
        # Transit probability (geometric probability of observing transit)
        # P_tr ∝ Rstar / a
        features['transit_probability'] = (
            features['stellar_radius_estimate'] / 
            features['semimajor_axis_estimate']
        )
        
        # Depth-to-noise ratio
        features['depth_noise_ratio'] = (
            features['transit_depth'] * features['snr']
        )
        
        # Duration anomaly (compared to expected for circular orbit)
        # Eccentric orbits show duration anomalies
        expected_duration = (
            features['orbital_period'] * 
            features['transit_probability'] * 0.1
        )
        features['duration_anomaly'] = np.abs(
            features['transit_duration'] / 24.0 - expected_duration
        ) / expected_duration
        
        # V-shaped vs U-shaped transit indicator
        # Related to impact parameter
        features['transit_shape_indicator'] = (
            features['duration_period_ratio'] / 
            np.sqrt(features['transit_depth'])
        )
        
        return features
    
    def calculate_statistical_features(self, df):
        """
        Calculate statistical features for classification.
        """
        features = df.copy()
        
        # Interaction terms
        features['period_snr_interaction'] = (
            np.log1p(features['orbital_period']) * 
            np.log1p(features['snr'])
        )
        
        features['depth_duration_interaction'] = (
            features['transit_depth'] * 
            features['transit_duration']
        )
        
        features['stellar_planet_interaction'] = (
            features['stellar_mass'] * 
            features['radius_ratio']
        )
        
        # Ratios and normalized features
        features['snr_per_depth'] = (
            features['snr'] / (features['transit_depth'] + 1e-6)
        )
        
        features['period_normalized_duration'] = (
            features['transit_duration'] / 
            np.log1p(features['orbital_period'])
        )
        
        return features
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding engineered features.
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[
                'orbital_period', 'transit_duration', 'transit_depth', 
                'snr', 'stellar_mass', 'stellar_temp', 'stellar_magnitude'
            ])
        
        # Create a copy to avoid modifying original data
        features = X.copy()
        
        # Apply all feature engineering steps
        features = self.calculate_transit_features(features)
        features = self.calculate_stellar_features(features)
        features = self.calculate_orbital_features(features)
        features = self.calculate_detection_features(features)
        features = self.calculate_statistical_features(features)
        
        # Handle any inf or nan values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        # Store feature names
        self.feature_names_ = features.columns.tolist()
        
        return features
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.feature_names_ if self.feature_names_ is not None else []


def calculate_planet_properties(orbital_period, transit_depth, stellar_mass, 
                                stellar_temp, stellar_magnitude, transit_duration):
    """
    Calculate physical planet properties from observational parameters.
    
    Returns:
        dict: Planet radius, temperature, semi-major axis, impact parameter
    """
    # Stellar radius estimate (solar radii)
    stellar_radius = stellar_mass ** 0.8 * (stellar_temp / 5778) ** 0.5
    
    # Planet radius from transit depth (Earth radii)
    radius_ratio = np.sqrt(transit_depth)
    planet_radius = radius_ratio * stellar_radius * 109.1  # Convert to Earth radii
    
    # Semi-major axis using Kepler's Third Law (AU)
    semi_major_axis = ((orbital_period / 365.25) ** (2/3) * 
                      stellar_mass ** (1/3))
    
    # Stellar luminosity (solar luminosities)
    stellar_luminosity = stellar_mass ** 3.5
    
    # Planet equilibrium temperature (Kelvin)
    planet_temp = stellar_temp * np.sqrt(stellar_radius / (2 * semi_major_axis))
    planet_temp *= 0.01 * 109.1  # Convert to realistic temperature
    
    # Impact parameter estimate
    # From transit duration and other parameters
    transit_duration_days = transit_duration / 24.0
    duration_ratio = transit_duration_days / orbital_period
    
    # Simplified impact parameter calculation
    impact_parameter = 1 - (duration_ratio / (2 * np.sqrt(transit_depth)))
    impact_parameter = max(0, min(0.99, impact_parameter))
    
    return {
        'planet_radius': float(planet_radius),
        'planet_temp': float(planet_temp),
        'semi_major_axis': float(semi_major_axis),
        'impact_parameter': float(impact_parameter)
    }


if __name__ == "__main__":
    # Test feature engineering
    from data_preprocessing import load_nasa_exoplanet_data
    
    df = load_nasa_exoplanet_data()
    feature_cols = ['orbital_period', 'transit_duration', 'transit_depth', 
                    'snr', 'stellar_mass', 'stellar_temp', 'stellar_magnitude']
    
    engineer = TransitFeatureEngineering()
    features = engineer.fit_transform(df[feature_cols])
    
    print(f"Original features: {len(feature_cols)}")
    print(f"Engineered features: {len(features.columns)}")
    print(f"\nNew features created:")
    new_features = [col for col in features.columns if col not in feature_cols]
    for feat in new_features[:10]:
        print(f"  - {feat}")
    print(f"  ... and {len(new_features) - 10} more")
