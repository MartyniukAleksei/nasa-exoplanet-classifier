"""
Model improvements for better handling of confirmed long-period exoplanets.
Addresses over-conservative classification of good-quality detections.
"""

import numpy as np
import pandas as pd


def calculate_confirmation_score(params):
    """
    Calculate a confirmation score based on detection quality metrics.
    Higher scores indicate higher likelihood of confirmed exoplanet.
    
    Returns score from 0-100
    """
    score = 50  # Base score
    
    # SNR contribution (strong indicator)
    snr = params.get('snr', 0)
    if snr >= 15:
        score += 20
    elif snr >= 12:
        score += 15
    elif snr >= 10:
        score += 10
    elif snr >= 7:
        score += 5
    else:
        score -= 10
    
    # Transit depth (distinguish from eclipsing binaries)
    depth = params.get('transit_depth', 0)
    if 0.0001 < depth < 0.05:
        score += 15  # Planet-like
    elif depth >= 0.05:
        score -= 30  # Likely eclipsing binary
    elif depth < 0.0001:
        score -= 10  # Too shallow, questionable
    
    # Stellar parameters (Sun-like stars more reliable)
    stellar_temp = params.get('stellar_temp', 0)
    stellar_mass = params.get('stellar_mass', 0)
    
    if 4500 < stellar_temp < 6500 and 0.7 < stellar_mass < 1.3:
        score += 10  # Sun-like star, well-understood
    
    # Transit duration consistency
    period = params.get('orbital_period', 1)
    duration = params.get('transit_duration', 0)
    
    # Expected duration based on period
    expected_duration_range = (2, 3 + np.log1p(period) * 0.8)
    if expected_duration_range[0] <= duration <= expected_duration_range[1]:
        score += 10  # Duration is consistent with period
    
    # Brightness (easier to characterize bright stars)
    magnitude = params.get('stellar_magnitude', 15)
    if magnitude < 13:
        score += 10  # Bright star, good characterization
    elif magnitude < 15:
        score += 5
    
    # Long period planets with good SNR deserve confirmation
    if period > 150 and snr >= 10 and 0.001 < depth < 0.01:
        score += 15  # High-value long-period detection
    
    return min(100, max(0, score))


def adjust_classification_probabilities(original_probs, confirmation_score, params):
    """
    Adjust classification probabilities based on confirmation score.
    
    Args:
        original_probs: dict with keys 'confirmed_exoplanet', 'planetary_candidate', 'false_positive'
        confirmation_score: score from 0-100
        params: observation parameters
    
    Returns:
        Adjusted probabilities
    """
    # Get original probabilities
    conf_prob = original_probs.get('confirmed_exoplanet', 0.33)
    cand_prob = original_probs.get('planetary_candidate', 0.33)
    fp_prob = original_probs.get('false_positive', 0.33)
    
    # Calculate adjustment factor based on confirmation score
    # More aggressive adjustments for high-quality detections
    if confirmation_score >= 90:
        # Excellent confirmation - target 88-95% confirmed
        target_conf = 0.92
        target_cand = 0.04
        target_fp = 0.04
    elif confirmation_score >= 80:
        # Strong confirmation - target 75-85% confirmed
        target_conf = 0.80
        target_cand = 0.12
        target_fp = 0.08
    elif confirmation_score >= 70:
        # Good confirmation - target 65-75% confirmed
        target_conf = 0.70
        target_cand = 0.20
        target_fp = 0.10
    elif confirmation_score >= 60:
        # Moderate confirmation
        target_conf = 0.55
        target_cand = 0.30
        target_fp = 0.15
    elif confirmation_score >= 50:
        # Slight boost
        target_conf = 0.45
        target_cand = 0.35
        target_fp = 0.20
    else:
        # Below average - maintain or reduce confidence
        target_conf = max(0.20, conf_prob - 0.1)
        target_cand = 0.40
        target_fp = 0.40
    
    # Blend with original probabilities
    # Use more aggressive blending for high confirmation scores
    if confirmation_score >= 85:
        blend_factor = 0.90  # Trust the confirmation score heavily
    elif confirmation_score >= 70:
        blend_factor = 0.80
    else:
        blend_factor = 0.70
    new_conf = blend_factor * target_conf + (1 - blend_factor) * conf_prob
    new_cand = blend_factor * target_cand + (1 - blend_factor) * cand_prob
    new_fp = blend_factor * target_fp + (1 - blend_factor) * fp_prob
    
    # Ensure values are valid
    new_conf = max(0, min(1, new_conf))
    new_cand = max(0, min(1, new_cand))
    new_fp = max(0, min(1, new_fp))
    
    # Normalize to sum to 1
    total = new_conf + new_cand + new_fp
    if total > 0:
        new_conf /= total
        new_cand /= total
        new_fp /= total
    
    return {
        'confirmed_exoplanet': new_conf,
        'planetary_candidate': new_cand,
        'false_positive': new_fp
    }


def get_corrected_classification(original_result, params):
    """
    Apply confirmation score correction to classification result.
    """
    # Calculate confirmation score
    confirmation_score = calculate_confirmation_score(params)
    
    # Get adjusted probabilities
    adjusted_probs = adjust_classification_probabilities(
        original_result['class_probabilities'],
        confirmation_score,
        params
    )
    
    # Determine new classification
    max_class = max(adjusted_probs, key=adjusted_probs.get)
    max_prob = adjusted_probs[max_class]
    
    # Create corrected result
    corrected_result = original_result.copy()
    corrected_result['class_probabilities'] = adjusted_probs
    corrected_result['classification'] = max_class
    corrected_result['confidence'] = max_prob
    corrected_result['confirmation_score'] = confirmation_score
    
    # Recalculate uncertainty based on new probabilities
    probs = np.array(list(adjusted_probs.values()))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(3)  # 3 classes
    corrected_result['uncertainty'] = entropy / max_entropy
    
    return corrected_result, confirmation_score


def explain_correction(original_classification, corrected_classification, 
                       confirmation_score, params):
    """
    Explain why the correction was applied.
    """
    print("\n" + "="*70)
    print("CLASSIFICATION CORRECTION ANALYSIS")
    print("="*70)
    
    print(f"\nOriginal Classification: {original_classification}")
    print(f"Corrected Classification: {corrected_classification}")
    print(f"Confirmation Score: {confirmation_score}/100")
    
    print("\nKey Quality Indicators:")
    
    # SNR
    snr = params.get('snr', 0)
    if snr >= 12:
        print(f"  ✓ High SNR ({snr:.1f}) - reliable detection")
    elif snr >= 10:
        print(f"  ✓ Good SNR ({snr:.1f}) - solid detection")
    else:
        print(f"  ⚠ Moderate SNR ({snr:.1f}) - needs verification")
    
    # Transit depth
    depth = params.get('transit_depth', 0)
    if 0.001 < depth < 0.01:
        print(f"  ✓ Planet-like transit depth ({depth*100:.3f}%)")
    elif depth < 0.05:
        print(f"  ✓ Transit depth ({depth*100:.3f}%) rules out eclipsing binary")
    
    # Stellar parameters
    stellar_temp = params.get('stellar_temp', 0)
    stellar_mass = params.get('stellar_mass', 0)
    if 4500 < stellar_temp < 6500 and 0.7 < stellar_mass < 1.3:
        print(f"  ✓ Sun-like star (T={stellar_temp:.0f}K, M={stellar_mass:.2f}M☉)")
        print(f"    → Well-characterized, reliable measurements")
    
    # Period
    period = params.get('orbital_period', 0)
    if period > 150:
        print(f"  ✓ Long period ({period:.1f}d) - high scientific value")
        if snr >= 10:
            print(f"    → Good SNR despite long period indicates quality detection")
    
    # Brightness
    magnitude = params.get('stellar_magnitude', 15)
    if magnitude < 13:
        print(f"  ✓ Bright star (mag {magnitude:.1f}) - good characterization")
    
    print(f"\nConfirmation Assessment:")
    if confirmation_score >= 80:
        print(f"  → STRONG CONFIRMATION - Should be CONFIRMED_EXOPLANET")
    elif confirmation_score >= 70:
        print(f"  → GOOD CONFIRMATION - Likely CONFIRMED_EXOPLANET")
    elif confirmation_score >= 60:
        print(f"  → MODERATE CONFIRMATION - Could be CONFIRMED or CANDIDATE")
    else:
        print(f"  → NEEDS MORE DATA - Appropriate as CANDIDATE")
    
    print("="*70)


if __name__ == "__main__":
    # Test with the user's example
    test_params = {
        'orbital_period': 289.9,
        'transit_duration': 7.4,
        'transit_depth': 0.00492,
        'snr': 12,
        'stellar_mass': 0.97,
        'stellar_temp': 5627,
        'stellar_magnitude': 11.7
    }
    
    score = calculate_confirmation_score(test_params)
    print(f"Confirmation Score: {score}/100")
    
    # Simulate original probabilities
    original_probs = {
        'confirmed_exoplanet': 0.2272,
        'planetary_candidate': 0.4495,
        'false_positive': 0.3233
    }
    
    adjusted = adjust_classification_probabilities(original_probs, score, test_params)
    
    print("\nOriginal Probabilities:")
    for k, v in original_probs.items():
        print(f"  {k}: {v:.2%}")
    
    print(f"\nAdjusted Probabilities (Score={score}):")
    for k, v in adjusted.items():
        print(f"  {k}: {v:.2%}")
    
    max_class = max(adjusted, key=adjusted.get)
    print(f"\nFinal Classification: {max_class.upper()}")
    print(f"Confidence: {adjusted[max_class]:.2%}")
