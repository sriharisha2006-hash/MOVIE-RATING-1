from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not found. Please run model.py first.'}), 500
    
    try:
        # Get data from form (Fetch API uses x-www-form-urlencoded here)
        budget = float(request.form.get('budget'))
        duration = float(request.form.get('duration'))
        
        # Prepare input for model
        input_features = np.array([[budget, duration]])
        
        # Predict
        prediction = model.predict(input_features)[0]
        
        # Clamp prediction between 0 and 10 and round to 1 decimal
        rating = round(min(max(prediction, 1.0), 10.0), 1)
        
        return jsonify({
            'success': True,
            'rating': f"{rating:.1f}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == "__main__":
    # Use host 0.0.0.0 and port 10000 for Render compatibility as requested
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
