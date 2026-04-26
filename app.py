from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import requests

app = Flask(__name__)

# --- CONFIGURATION ---
OMDB_API_KEY = "f26d59c5" 
MODEL_PATH = 'model.pkl'

# --- VIP DATABASE (Priority over API) ---
# We add this because sometimes OMDb search returns the wrong movie for short titles
VIP_RATINGS = {
    "kgf": {
        "rating": "8.2", # Official IMDb is 8.2, but I'll use 8.1 if you prefer
        "poster": "https://m.media-amazon.com/images/M/MV5BM2M0YmIxNzItOWI4My00MmQzLWE0NGYtZTM3NjllNjIwZjc5XkEyXkFqcGc@._V1_SX300.jpg",
        "genre": "Action, Crime, Drama",
        "year": "2018"
    },
    "kgf 1": {
        "rating": "8.2",
        "poster": "https://m.media-amazon.com/images/M/MV5BM2M0YmIxNzItOWI4My00MmQzLWE0NGYtZTM3NjllNjIwZjc5XkEyXkFqcGc@._V1_SX300.jpg",
        "genre": "Action, Crime, Drama",
        "year": "2018"
    },
    "kgf chapter 1": {
        "rating": "8.2",
        "poster": "https://m.media-amazon.com/images/M/MV5BM2M0YmIxNzItOWI4My00MmQzLWE0NGYtZTM3NjllNjIwZjc5XkEyXkFqcGc@._V1_SX300.jpg",
        "genre": "Action, Crime, Drama",
        "year": "2018"
    }
}

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

def get_real_imdb_data(title):
    """Fetch real rating and poster from OMDb API"""
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if data.get("Response") == "True" and data.get("imdbRating") != "N/A":
            return {
                "rating": data.get("imdbRating"),
                "poster": data.get("Poster"),
                "year": data.get("Year"),
                "genre": data.get("Genre"),
                "found": True
            }
    except Exception as e:
        print(f"API Error: {e}")
    return {"found": False}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_name = request.form.get('name', 'Unknown Movie')
        name_lower = raw_name.lower().strip()
        
        # 1. Check VIP Database FIRST (High Priority)
        if name_lower in VIP_RATINGS:
            vip = VIP_RATINGS[name_lower]
            return jsonify({
                'success': True,
                'name': raw_name,
                'genre': vip["genre"],
                'year': vip["year"],
                'rating': vip["rating"],
                'is_real': True,
                'poster': vip["poster"]
            })
        
        # 2. Try OMDb API
        movie_data = get_real_imdb_data(raw_name)
        if movie_data["found"]:
            return jsonify({
                'success': True,
                'name': raw_name,
                'genre': movie_data["genre"],
                'year': movie_data["year"],
                'rating': movie_data["rating"],
                'is_real': True,
                'poster': movie_data["poster"] if movie_data["poster"] != "N/A" else None
            })
        
        # 3. Fallback to AI Prediction
        budget = float(request.form.get('budget', 0))
        duration = float(request.form.get('duration', 0))
        if model:
            input_features = np.array([[budget, duration]])
            prediction = model.predict(input_features)[0]
            rating = round(min(max(prediction, 1.0), 10.0), 1)
        else:
            rating = "7.0"

        return jsonify({
            'success': True,
            'name': raw_name,
            'genre': request.form.get('genre'),
            'year': request.form.get('year'),
            'rating': f"{rating}",
            'is_real': False,
            'poster': None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
