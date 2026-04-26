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
        return jsonify({'success': False, 'error': 'Model not found.'}), 500
    
    try:
        # Get metadata
        name = request.form.get('name', 'Unknown Movie')
        genre = request.form.get('genre', 'General')
        year = request.form.get('year', '2024')
        
        # Get numeric features for ML model
        budget = float(request.form.get('budget', 0))
        duration = float(request.form.get('duration', 0))
        
        # Predict Rating
        input_features = np.array([[budget, duration]])
        prediction = model.predict(input_features)[0]
        rating = round(min(max(prediction, 1.0), 10.0), 1)
        
        return jsonify({
            'success': True,
            'name': name,
            'genre': genre,
            'year': year,
            'duration': duration,
            'rating': f"{rating:.1f}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
