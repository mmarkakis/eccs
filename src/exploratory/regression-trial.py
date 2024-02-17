# Need to think a lot more carefully what LASSO gives us

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import chain, combinations
import matplotlib.pyplot as plt
import seaborn as sns
# Function to calculate the average treatment effect
def calculate_ate(model, X_test, subset_condition):
    subset_indices = subset_condition(X_test)
    X_subset = X_test[subset_indices]
    
    # Make predictions on the subset
    y_subset_pred = model.predict(X_subset[:, subset])
    
    # Calculate the average predicted value for the subset
    average_predicted_value = np.mean(y_subset_pred)
    
    return average_predicted_value

# Function to partition the test set based on X2
def partition_condition(X_test, threshold):
    return X_test[:, 1] >= threshold  # Assuming X2 is at index 1


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
X_test_lt_05 = X_test[X_test[:, 1] < 0.5]
X_test_gte_05 = X_test[X_test[:, 1] >= 0.5]

# Fit the model with all possible subsets and select the best one based on mean squared error
best_subset = None
best_model = None
best_mse = float('inf')
best_lt = None
best_gte = None
biggest_dif = 0
biggest_dif_set = None
ATEs = []

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
    ate_lt_05 = calculate_ate(model, X_test_lt_05, lambda x: x[:, 1] < 0.5)
    ate_gte_05 = calculate_ate(model, X_test_gte_05, lambda x: x[:, 1] >= 0.5)

    # Update the best model if the current one has lower MSE
    print("Subset:", subset)
    print("Coefficients:", model.coef_)
    print("MSE:", mse)
    print('Average Treatment Effect (X2 < 0.5):', ate_lt_05 - ate_gte_05)
    ATEs.append(ate_lt_05 - ate_gte_05)
    if mse < best_mse:
        best_mse = mse
        best_subset = subset
        best_model = model
        best_lt = ate_lt_05
        best_gte = ate_gte_05
    if abs(ate_lt_05 - ate_gte_05) > biggest_dif:
        biggest_dif = abs(ate_lt_05 - ate_gte_05)
        biggest_dif_set = subset


# Print the best subset and coefficients
print('Best Subset:', best_subset)
print('Best Coefficients:', best_model.coef_)
print('Intercept:', best_model.intercept_)
print('Best Mean Squared Error:', best_mse)
print('Biggest ATE', biggest_dif)
print('attained by', biggest_dif_set)
plt.figure(figsize=(8, 6))
plt.hist(ATEs, bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Floats')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Seaborn KDE plot
plt.figure(figsize=(8, 6))
sns.kdeplot(ATEs, color='green', fill=True)
plt.title('Kernel Density Estimation (KDE) Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()