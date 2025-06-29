import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump

# Generate synthetic login data
np.random.seed(42)
N = 1000
# Features: hour (0-23), geo_distance (km), failed_attempts, api_rate
hours = np.random.randint(0, 24, N)
distances = np.random.exponential(scale=50, size=N)  # Most logins are close, some far
failed_attempts = np.random.poisson(1, N)
api_rates = np.random.poisson(20, N)

# Add some anomalies
for i in range(30):
    hours[i] = np.random.choice([2, 3, 4, 23])  # Odd hours
    distances[i] = np.random.uniform(500, 2000)  # Far away
    failed_attempts[i] = np.random.randint(5, 10)
    api_rates[i] = np.random.randint(100, 200)

X = np.column_stack([hours, distances, failed_attempts, api_rates])

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

dump(model, 'trust_model.joblib')
print('Isolation Forest model trained and saved as trust_model.joblib') 