"""
Flask API for Exoplanet Classification System
Receives requests from Heroku website and returns predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from exoplanet_classifier import ExoplanetClassificationSystem
import os

app = Flask(__name__)

# Enable CORS for your GitHub Pages site
CORS(app, 
     origins=[
         "https://martyniukaleksei.github.io",
         "http://localhost:*",
         "http://127.0.0.1:*"
     ],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Accept"],
     supports_credentials=False,
     max_age=3600)

# Load the model once when server starts
print("Loading exoplanet classification model...")
system = ExoplanetClassificationSystem()
system.load_models()
print("Model loaded successfully!")


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Exoplanet Classification API',
        'endpoints': {
            '/predict': 'POST - Make a prediction',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': system.is_trained
    })


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Handle preflight OPTIONS request"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return predict_exoplanet()


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle preflight OPTIONS request"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return predict_exoplanet()


def predict_exoplanet():
    """
    Main prediction function used by both /analyze and /predict endpoints
    
    Expects JSON:
    {
        "orbital_period": 289.9,
        "transit_duration": 7.4,
        "transit_depth": 0.00492,
        "snr": 12,
        "stellar_mass": 0.97,
        "stellar_temp": 5627,
        "stellar_magnitude": 11.7
    }
    
    Returns JSON:
    {
        "classification": "confirmed_exoplanet",
        "confidence": 0.8570,
        "properties": {
            "planet_radius": 7.8106,
            "planet_temp": 43.5868,
            "semi_major_axis": 0.8522,
            "impact_parameter": 0.4699
        }
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'orbital_period',
            'transit_duration',
            'transit_depth',
            'snr',
            'stellar_mass',
            'stellar_temp',
            'stellar_magnitude'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Create input dictionary
        input_features = {
            'orbital_period': float(data['orbital_period']),
            'transit_duration': float(data['transit_duration']),
            'transit_depth': float(data['transit_depth']),
            'snr': float(data['snr']),
            'stellar_mass': float(data['stellar_mass']),
            'stellar_temp': float(data['stellar_temp']),
            'stellar_magnitude': float(data['stellar_magnitude'])
        }
        
        # Make prediction
        result = system.predict(input_features, return_uncertainty=True)
        
        # Format response
        response = {
            'status': 'success',
            'classification': result['classification'],
            'confidence': float(result['confidence']),
            'class_probabilities': {
                k: float(v) for k, v in result['class_probabilities'].items()
            }
        }
        
        # Add planet properties ONLY if they exist and classification is planet-related
        if 'properties' in result and result['properties'] is not None:
            # Only return properties for confirmed exoplanets and candidates
            if result['classification'] in ['confirmed_exoplanet', 'planetary_candidate']:
                response['properties'] = {
                    'planet_radius': float(result['properties']['planet_radius']),
                    'planet_temp': float(result['properties']['planet_temp']),
                    'semi_major_axis': float(result['properties']['semi_major_axis']),
                    'impact_parameter': float(result['properties']['impact_parameter'])
                }
            # For false positives, explicitly set properties to null
            else:
                response['properties'] = None
                response['message'] = 'No planetary properties - classified as false positive'
        
        # Add uncertainty if available
        if 'uncertainty' in result:
            response['uncertainty'] = float(result['uncertainty'])
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'error': f'Invalid input values: {str(e)}'
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expects JSON:
    {
        "observations": [
            {
                "orbital_period": 289.9,
                "transit_duration": 7.4,
                ...
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'observations' not in data:
            return jsonify({
                'error': 'Missing "observations" array in request'
            }), 400
        
        observations = data['observations']
        results = []
        
        for i, obs in enumerate(observations):
            try:
                input_features = {
                    'orbital_period': float(obs['orbital_period']),
                    'transit_duration': float(obs['transit_duration']),
                    'transit_depth': float(obs['transit_depth']),
                    'snr': float(obs['snr']),
                    'stellar_mass': float(obs['stellar_mass']),
                    'stellar_temp': float(obs['stellar_temp']),
                    'stellar_magnitude': float(obs['stellar_magnitude'])
                }
                
                result = system.predict(input_features)
                
                prediction = {
                    'index': i,
                    'status': 'success',
                    'classification': result['classification'],
                    'confidence': float(result['confidence'])
                }
                
                if 'properties' in result:
                    prediction['properties'] = {
                        k: float(v) for k, v in result['properties'].items()
                    }
                
                results.append(prediction)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'total': len(observations),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)