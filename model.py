import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    # 1. Create a Realistic IMDb-style dataset
    # We want to mimic the "IMDb curve" where most movies are 6.0-8.5
    np.random.seed(42)
    n_samples = 500
    
    # Randomly generate Budget (5M to 300M) and Duration (80 to 180m)
    budgets = np.random.uniform(5, 300, n_samples)
    durations = np.random.uniform(80, 180, n_samples)
    
    # Formula for "Realistic" IMDb Rating:
    # Baseline 6.0 + small boost for budget + small boost for duration + bell-curve noise
    # This prevents the AI from giving extreme 1.0 or 10.0 ratings too easily
    ratings = 5.8 + (0.003 * budgets) + (0.008 * durations) + np.random.normal(0, 0.4, n_samples)
    
    # Clip ratings to 1.0 - 9.5 (Hardly anything is 10.0 on IMDb)
    ratings = np.clip(ratings, 1.0, 9.5)
    
    data = pd.DataFrame({
        'Budget': budgets,
        'Duration': durations,
        'Rating': ratings
    })
    
    print("Realistic Dataset Sample (First 5):")
    print(data.head())
    
    # 2. Train the Model
    X = data[['Budget', 'Duration']]
    y = data['Rating']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("\n[SUCCESS] Model retrained with IMDb-style logic!")
    print(f"Typical Rating for 100M Budget / 120m Movie: {model.predict([[100, 120]])[0]:.1f}")
    
    # 3. Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved as 'model.pkl'")

if __name__ == "__main__":
    train_model()
