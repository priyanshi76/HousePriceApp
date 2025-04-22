import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv('housing.csv')

# Example: Area and Bedrooms as features
X = df[['area', 'bedrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
