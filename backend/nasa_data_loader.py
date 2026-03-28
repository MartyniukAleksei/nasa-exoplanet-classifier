"""
Load real confirmed exoplanet data from NASA Exoplanet Archive.
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO


def download_nasa_confirmed_exoplanets():
    """
    Download confirmed exoplanet data from NASA Exoplanet Archive.
    """
    print("Downloading confirmed exoplanet data from NASA Exoplanet Archive...")
    
    # NASA Exoplanet Archive TAP service
    # Get confirmed exoplanets with transit data
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Query for confirmed transiting exoplanets
    query = """
    SELECT 
        pl_name,
        pl_orbper as orbital_period,
        pl_trandur as transit_duration,
        pl_trandep as transit_depth,
        pl_ratror as planet_radius_ratio,
        pl_rade as planet_radius,
        pl_orbsmax as semi_major_axis,
        pl_eqt as planet_temp,
        pl_imppar as impact_parameter,
        st_mass as stellar_mass,
        st_teff as stellar_temp,
        st_optmag as stellar_magnitude,
        sy_snum,
        sy_pnum
    FROM ps 
    WHERE 
        pl_orbper IS NOT NULL 
        AND pl_trandur IS NOT NULL
        AND pl_trandep IS NOT NULL
        AND st_mass IS NOT NULL
        AND st_teff IS NOT NULL
        AND pl_rade IS NOT NULL
        AND default_flag = 1
    """
    
    params = {
        'request': 'doQuery',
        'lang': 'ADQL',
        'query': query,
        'format': 'csv'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            print(f"✓ Downloaded {len(df)} confirmed exoplanets from NASA")
            return df
        else:
            print(f"Failed to download: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading NASA data: {e}")
        return None


def create_training_dataset_from_nasa(nasa_df, n_target=100):
    """
    Create training dataset from NASA confirmed exoplanets.
    """
    print(f"\nProcessing NASA data for training...")
    
    # Clean the NASA data
    df_clean = nasa_df.copy()
    
    # Remove any remaining NaNs in critical columns
    required_cols = ['orbital_period', 'transit_duration', 'transit_depth', 
                     'stellar_mass', 'stellar_temp']
    df_clean = df_clean.dropna(subset=required_cols)
    
    # Convert transit depth to decimal if needed (some are in %)
    if df_clean['transit_depth'].max() > 1:
        df_clean['transit_depth'] = df_clean['transit_depth'] / 100.0
    
    # Convert transit duration from hours if needed
    if df_clean['transit_duration'].median() > 24:
        df_clean['transit_duration'] = df_clean['transit_duration'] / 60.0  # minutes to hours
    
    # Estimate SNR based on transit depth and other factors
    # Higher SNR for: brighter stars, deeper transits, well-studied systems
    base_snr = 15
    depth_factor = np.sqrt(df_clean['transit_depth'] * 1000)  # Deeper = easier to detect
    
    # Magnitude effect (brighter = higher SNR)
    mag_factor = np.where(
        df_clean['stellar_magnitude'].notna(),
        np.exp(-(df_clean['stellar_magnitude'] - 10) / 5),
        1.0
    )
    
    # Period effect (longer period = fewer transits = lower SNR)
    period_factor = 1.0 / (1 + df_clean['orbital_period'] / 50)
    
    df_clean['snr'] = base_snr * depth_factor * mag_factor * period_factor
    df_clean['snr'] = df_clean['snr'].clip(5, 50)  # Reasonable SNR range
    
    # Add some realistic noise
    df_clean['snr'] = df_clean['snr'] * np.random.uniform(0.8, 1.2, len(df_clean))
    
    # Handle missing stellar magnitude
    df_clean['stellar_magnitude'] = df_clean['stellar_magnitude'].fillna(14.0)
    
    # Select up to n_target confirmed exoplanets
    if len(df_clean) > n_target:
        # Prioritize diverse period range
        df_clean = df_clean.sort_values('orbital_period')
        indices = np.linspace(0, len(df_clean)-1, n_target, dtype=int)
        df_confirmed = df_clean.iloc[indices].copy()
    else:
        df_confirmed = df_clean.copy()
    
    # Add classification label
    df_confirmed['classification'] = 'confirmed_exoplanet'
    
    print(f"✓ Selected {len(df_confirmed)} confirmed exoplanets for training")
    print(f"  Period range: {df_confirmed['orbital_period'].min():.1f} - {df_confirmed['orbital_period'].max():.1f} days")
    print(f"  Median period: {df_confirmed['orbital_period'].median():.1f} days")
    
    return df_confirmed


def create_false_positives(n_samples=50):
    """
    Create realistic false positive examples.
    """
    print(f"\nGenerating {n_samples} false positive examples...")
    
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        fp_type = np.random.choice(['eclipsing_binary', 'background', 'artifact'], 
                                   p=[0.7, 0.2, 0.1])
        
        if fp_type == 'eclipsing_binary':
            # Eclipsing binary stars - MUCH deeper transits
            orbital_period = np.random.lognormal(0.5, 1.5)  # Short periods
            transit_depth = np.random.uniform(0.05, 0.30)  # 5-30% depth!
            transit_duration = np.random.normal(4, 1.5)
            snr = np.random.lognormal(2.5, 0.5)
            stellar_mass = np.random.normal(1.0, 0.5)
            stellar_temp = np.random.normal(5500, 1000)
            stellar_magnitude = np.random.normal(14, 2)
            
        elif fp_type == 'background':
            # Background eclipsing binary
            orbital_period = np.random.lognormal(1.5, 2.0)
            transit_depth = np.random.lognormal(-6, 1.0)  # Very shallow
            transit_duration = np.random.normal(3, 2)
            snr = np.random.lognormal(1.0, 1.0)  # Low SNR
            stellar_mass = np.random.normal(1.0, 0.5)
            stellar_temp = np.random.normal(5500, 1200)
            stellar_magnitude = np.random.normal(16, 2)
            
        else:  # artifact
            orbital_period = np.random.uniform(0.5, 50)
            transit_depth = np.random.lognormal(-6, 2.0)
            transit_duration = np.random.uniform(0.5, 8)
            snr = np.random.lognormal(0.5, 1.5)
            stellar_mass = np.random.normal(1.0, 0.5)
            stellar_temp = np.random.normal(5500, 1500)
            stellar_magnitude = np.random.normal(16, 3)
        
        data.append({
            'orbital_period': max(0.5, orbital_period),
            'transit_duration': max(0.5, transit_duration),
            'transit_depth': max(0.0001, min(0.5, transit_depth)),
            'snr': max(3, snr),
            'stellar_mass': max(0.1, stellar_mass),
            'stellar_temp': max(3000, min(10000, stellar_temp)),
            'stellar_magnitude': max(8, min(20, stellar_magnitude)),
            'classification': 'false_positive',
            'planet_radius': np.nan,
            'planet_temp': np.nan,
            'semi_major_axis': np.nan,
            'impact_parameter': np.nan
        })
    
    return pd.DataFrame(data)


def create_candidates(n_samples=30):
    """
    Create planetary candidate examples (lower SNR, less certain).
    """
    print(f"\nGenerating {n_samples} planetary candidate examples...")
    
    np.random.seed(43)
    data = []
    
    for i in range(n_samples):
        # Similar to confirmed but lower quality
        period_type = np.random.choice(['short', 'medium', 'long'], p=[0.3, 0.4, 0.3])
        
        if period_type == 'short':
            orbital_period = np.random.lognormal(1.5, 1.2)
        elif period_type == 'medium':
            orbital_period = np.random.lognormal(3.5, 1.0)
        else:
            orbital_period = np.random.lognormal(5.0, 0.8)
        
        stellar_mass = np.random.normal(1.0, 0.4)
        stellar_temp = np.random.normal(5500, 1000)
        stellar_magnitude = np.random.normal(15, 2)
        
        planet_radius = np.random.lognormal(0.5, 1.0)
        stellar_radius = abs(stellar_mass) ** 0.8
        transit_depth = (planet_radius / (stellar_radius * 109.1)) ** 2
        
        base_duration = 2 + np.log1p(abs(orbital_period)) * 0.5
        transit_duration = abs(np.random.normal(base_duration, base_duration * 0.3))
        
        # Lower SNR for candidates
        snr_mean = max(4, 10 / (1 + abs(orbital_period) / 50))
        snr = abs(np.random.lognormal(np.log(snr_mean), 0.7))
        
        semi_major_axis = (orbital_period / 365.25) ** (2/3) * stellar_mass ** (1/3)
        planet_temp = stellar_temp * np.sqrt(1 / (2 * semi_major_axis)) * 0.01
        impact_parameter = np.random.uniform(0, 0.95)
        
        data.append({
            'orbital_period': max(0.5, orbital_period),
            'transit_duration': max(0.5, transit_duration),
            'transit_depth': max(0.0001, min(0.1, transit_depth)),
            'snr': max(4, snr),
            'stellar_mass': max(0.1, stellar_mass),
            'stellar_temp': max(3000, min(10000, stellar_temp)),
            'stellar_magnitude': max(8, min(20, stellar_magnitude)),
            'classification': 'planetary_candidate',
            'planet_radius': planet_radius,
            'planet_temp': planet_temp,
            'semi_major_axis': semi_major_axis,
            'impact_parameter': impact_parameter
        })
    
    return pd.DataFrame(data)


def create_complete_training_dataset():
    """
    Create complete training dataset with NASA confirmed exoplanets.
    """
    print("="*70)
    print("CREATING TRAINING DATASET WITH NASA CONFIRMED EXOPLANETS")
    print("="*70)
    
    # Download NASA data
    nasa_df = download_nasa_confirmed_exoplanets()
    
    if nasa_df is None or len(nasa_df) < 50:
        print("\n⚠️  Could not download NASA data or insufficient data")
        print("Using enhanced synthetic data instead...")
        from data_preprocessing import load_nasa_exoplanet_data
        return load_nasa_exoplanet_data()
    
    # Process NASA data
    df_confirmed = create_training_dataset_from_nasa(nasa_df, n_target=100)
    
    # Create false positives and candidates
    df_fp = create_false_positives(n_samples=50)
    df_candidates = create_candidates(n_samples=30)
    
    # Combine all data
    df_all = pd.concat([df_confirmed, df_fp, df_candidates], ignore_index=True)
    
    # Shuffle
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df_all)}")
    print(f"\nClass distribution:")
    print(df_all['classification'].value_counts())
    print(f"\nConfirmed exoplanet period range:")
    confirmed = df_all[df_all['classification'] == 'confirmed_exoplanet']
    print(f"  Min: {confirmed['orbital_period'].min():.1f} days")
    print(f"  Median: {confirmed['orbital_period'].median():.1f} days")
    print(f"  Max: {confirmed['orbital_period'].max():.1f} days")
    print("="*70)
    
    return df_all


if __name__ == "__main__":
    df = create_complete_training_dataset()
    
    # Save for inspection
    df.to_csv('nasa_training_data.csv', index=False)
    print("\n✓ Training data saved to nasa_training_data.csv")
