# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Display the available features to the user
print("Available Features:")
print("\n".join(data.columns))

# Preprocessing: Convert boolean values to 0 and 1
bool_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
data[bool_cols] = data[bool_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Convert furnishingstatus to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

# Let the user choose features
selected_features = input("Enter the features you want to use for prediction (comma-separated): ")
selected_features = selected_features.split(',')

# Define X (features) and y (target)
X = data[selected_features]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display R2 and RMSE scores
print(f'R2 Score: {r2}')
print(f'RMSE: {rmse}')

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
