import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Numerically stable sigmoid activation function."""
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))

    def initialize_parameters(self, num_features):
        """Initialize weights and bias."""
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def compute_cost(self, y, y_pred):
        """Calculate the binary cross-entropy cost function."""
        m = y.shape[0]
        # Clamp y_pred to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -(1 / m) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        return cost

    def propagate(self, X, y):
        """Forward and backward propagation step."""
        # Forward propagation
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        
        # Compute cost
        cost = self.compute_cost(y, y_pred)
        
        # Backward propagation (gradient computation)
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        
        return dw, db, cost

    def fit(self, X, y):
        """Train the logistic regression model."""
        # Initialize parameters
        num_features = X.shape[1]
        self.initialize_parameters(num_features)
        
        # Gradient descent
        for i in range(self.num_iterations):
            dw, db, cost = self.propagate(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")
    
    def predict(self, X):
        """Make predictions with the trained model."""
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred > 0.5

# Example usage:

df = pd.read_csv('HeartDisease.csv')

X = df[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']]
y = df[['HeartDisease']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = LogisticRegression(learning_rate=0.01, num_iterations=300000)
print(X_train)  
print(y_train)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
