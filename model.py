import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    # 1. Create a synthetic dataset
    # Budget (in millions), Duration (in minutes)
    # Target: Rating (out of 10)
    # Simple logic: Rating increases slightly with duration and budget, but with noise
    np.random.seed(42)
    n_samples = 100
    
    budgets = np.random.uniform(5, 300, n_samples)  # 5M to 300M
    durations = np.random.uniform(80, 180, n_samples) # 80 to 180 mins
    
    # Simple linear relationship + some noise
    # Rating = 5 + 0.005*Budget + 0.01*Duration + noise
    ratings = 4 + (0.005 * budgets) + (0.012 * durations) + np.random.normal(0, 0.5, n_samples)
    ratings = np.clip(ratings, 1, 10) # Keep rating between 1 and 10
    
    data = pd.DataFrame({
        'Budget': budgets,
        'Duration': durations,
        'Rating': ratings
    })
    
    print("Dataset Sample:")
    print(data.head())
    
    # 2. Train Linear Regression Model
    X = data[['Budget', 'Duration']]
    y = data['Rating']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("\nModel trained successfully!")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    
    # 3. Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\nModel saved as 'model.pkl'")

if __name__ == "__main__":
    train_model()
