# Need to think a lot more carefully what LASSO gives us

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import chain, combinations

# Function to generate all subsets of a set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# Set a random seed for reproducibility
np.random.seed(42)

# Generate a dataset with 10 dimensions
num_samples = 100
num_dimensions = 10

# Create random values for x1 to x8
X = np.random.rand(num_samples, num_dimensions - 1)

# Create x9 with values centered around 0
X9 = np.random.normal(loc=0, scale=0.1, size=(num_samples, 1))

# Concatenate x1 to x8 with x9
X = np.concatenate((X, X9), axis=1)

# Generate random values for y
y = 2 * X[:, 0] + 1.5 * X[:, 1] - 1.2 * X[:, 2] + 0.8 * X[:, 3] + 0.5 * X[:, 4] - 0.3 * X[:, 5] + 0.2 * X[:, 6] - 0.1 * X[:, 7] + 0.01 * X[:, 8] + np.random.normal(loc=0, scale=0.1, size=num_samples)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model with all possible subsets and select the best one based on mean squared error
best_subset = None
best_model = None
best_mse = float('inf')

for subset in powerset(range(num_dimensions - 1)):
    if len(subset) == 0:
        continue

    # Create a subset of X based on the current combination
    X_subset = X_train[:, subset]

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_subset, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test[:, subset])

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Update the best model if the current one has lower MSE
    print("Subset:", subset)
    print("Coefficients:", model.coef_)
    print("MSE:", mse)
    if mse < best_mse:
        best_mse = mse
        best_subset = subset
        best_model = model

# Print the best subset and coefficients
print('Best Subset:', best_subset)
print('Best Coefficients:', best_model.coef_)
print('Intercept:', best_model.intercept_)
print('Best Mean Squared Error:', best_mse)