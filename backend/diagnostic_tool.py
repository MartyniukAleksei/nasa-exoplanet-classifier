#!/usr/bin/env python3
"""
Diagnostic tool to explain exoplanet classifications.
"""

import numpy as np
import pandas as pd


def analyze_observation(params):
    """
    Analyze observation parameters and provide diagnostic insights.
    """
    print("\n" + "="*70)
    print("OBSERVATION DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # Extract parameters
    period = params['orbital_period']
    duration = params['transit_duration']
    depth = params['transit_depth']
    snr = params['snr']
    stellar_mass = params['stellar_mass']
    stellar_temp = params['stellar_temp']
    stellar_mag = params['stellar_magnitude']
    
    # Calculate derived quantities
    stellar_radius = stellar_mass ** 0.8  # Solar radii
    radius_ratio = np.sqrt(depth)
    planet_radius_est = radius_ratio * stellar_radius * 109.1  # Earth radii
    semi_major_axis = (period / 365.25) ** (2/3) * stellar_mass ** (1/3)  # AU
    
    print(f"\nInput Parameters:")
    print(f"  Orbital Period: {period:.1f} days")
    print(f"  Transit Duration: {duration:.1f} hours")
    print(f"  Transit Depth: {depth:.5f} ({depth*100:.3f}%)")
    print(f"  SNR: {snr:.1f}")
    print(f"  Stellar Mass: {stellar_mass:.2f} M☉")
    print(f"  Stellar Temp: {stellar_temp:.0f} K")
    print(f"  Stellar Magnitude: {stellar_mag:.1f}")
    
    print(f"\nDerived Properties:")
    print(f"  Estimated Planet Radius: {planet_radius_est:.2f} R⊕")
    print(f"  Estimated Orbital Distance: {semi_major_axis:.3f} AU")
    
    # Planet size classification
    if planet_radius_est < 0.5:
        size_class = "Sub-Earth (very small)"
    elif planet_radius_est < 2:
        size_class = "Earth to Super-Earth"
    elif planet_radius_est < 6:
        size_class = "Neptune-like"
    elif planet_radius_est < 15:
        size_class = "Jupiter-like (gas giant)"
    else:
        size_class = "Super-Jupiter (unlikely for planet)"
    
    print(f"  Size Classification: {size_class}")
    
    # Orbital classification
    if semi_major_axis < 0.1:
        orbit_class = "Very close orbit (Hot Jupiter/Neptune)"
    elif semi_major_axis < 0.5:
        orbit_class = "Close orbit (like Mercury)"
    elif semi_major_axis < 1.5:
        orbit_class = "Inner system (like Venus/Earth)"
    elif semi_major_axis < 5:
        orbit_class = "Outer system (like Mars/Jupiter)"
    else:
        orbit_class = "Far orbit"
    
    print(f"  Orbital Classification: {orbit_class}")
    
    # Detection quality
    print(f"\nDetection Quality Assessment:")
    
    if snr < 7:
        print(f"  ⚠️  SNR ({snr:.1f}) is below typical threshold (7)")
        print(f"      → Lower confidence, needs confirmation")
    elif snr < 12:
        print(f"  ✓  SNR ({snr:.1f}) is moderate")
        print(f"      → Reasonable detection quality")
    else:
        print(f"  ✓✓ SNR ({snr:.1f}) is good")
        print(f"      → High detection quality")
    
    # Period considerations
    if period > 200:
        print(f"  ℹ️  Long period ({period:.1f} days)")
        print(f"      → Fewer transits observed, harder to confirm")
        print(f"      → More likely to be classified as candidate")
    elif period < 2:
        print(f"  ℹ️  Very short period ({period:.1f} days)")
        print(f"      → Check for eclipsing binary")
    
    # Transit depth considerations
    if depth > 0.05:
        print(f"  ⚠️  Very deep transit ({depth*100:.2f}%)")
        print(f"      → Suggests very large object")
        print(f"      → Could be eclipsing binary star system")
    elif depth < 0.0001:
        print(f"  ⚠️  Very shallow transit ({depth*100:.4f}%)")
        print(f"      → Small planet or noisy signal")
        print(f"      → Needs high SNR for confirmation")
    else:
        print(f"  ✓  Transit depth ({depth*100:.3f}%) is planet-like")
    
    # Duration/Period ratio
    duration_days = duration / 24.0
    duration_ratio = duration_days / period
    
    if duration_ratio > 0.2:
        print(f"  ⚠️  Transit duration is very long relative to period")
        print(f"      → Unusual geometry, check for false positive")
    elif duration_ratio < 0.01:
        print(f"  ⚠️  Transit duration is very short relative to period")
        print(f"      → Grazing transit or unusual geometry")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    
    planet_indicators = 0
    fp_indicators = 0
    
    # Positive indicators
    if 0.0001 < depth < 0.05:
        planet_indicators += 1
    if 7 < snr < 100:
        planet_indicators += 1
    if 0.5 < planet_radius_est < 20:
        planet_indicators += 1
    if 0.01 < duration_ratio < 0.15:
        planet_indicators += 1
    if 3000 < stellar_temp < 8000:
        planet_indicators += 1
    if 0.5 < stellar_mass < 2.0:
        planet_indicators += 1
    
    # Negative indicators
    if depth > 0.05:
        fp_indicators += 1
    if snr < 7:
        fp_indicators += 1
    if planet_radius_est > 20:
        fp_indicators += 1
    if duration_ratio > 0.2:
        fp_indicators += 1
    
    print(f"  Planet-like indicators: {planet_indicators}/6")
    print(f"  False positive indicators: {fp_indicators}/4")
    
    if planet_indicators >= 5 and fp_indicators == 0:
        print(f"\n  → Strong planet candidate")
        print(f"     Expected: CONFIRMED_EXOPLANET")
    elif planet_indicators >= 3 and fp_indicators <= 1:
        print(f"\n  → Good planet candidate")
        print(f"     Expected: CONFIRMED_EXOPLANET or PLANETARY_CANDIDATE")
    elif planet_indicators >= 2:
        print(f"\n  → Moderate planet candidate")
        print(f"     Expected: PLANETARY_CANDIDATE")
    else:
        print(f"\n  → Uncertain or likely false positive")
        print(f"     Expected: PLANETARY_CANDIDATE or FALSE_POSITIVE")
    
    # Special notes
    print(f"\nSpecial Considerations:")
    if period > 200:
        print(f"  • Long-period planets are harder to detect and confirm")
        print(f"  • Fewer transits = lower SNR = higher uncertainty")
        print(f"  • Model may be less confident due to limited training data")
    
    if 0.8 < semi_major_axis < 1.2 and 200 < period < 400:
        print(f"  • This is in the 'habitable zone' range!")
        print(f"  • Similar to Venus/Earth orbital distance")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Example usage
    test_case = {
        'orbital_period': 289.9,
        'transit_duration': 7.4,
        'transit_depth': 0.00492,
        'snr': 12,
        'stellar_mass': 0.97,
        'stellar_temp': 5627,
        'stellar_magnitude': 11.7
    }
    
    analyze_observation(test_case)
