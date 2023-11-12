# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('IRIS.csv')

# Extract features and target variable
X = df.drop('species', axis=1)
y = df['species']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier with feature names
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Show progress using tqdm
print("Training and evaluation progress:")
for i in tqdm(range(5), desc="Processing", unit="step"):
    # Perform some intermediate steps (you can replace this with your actual steps)
    pass

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Display the results
print(f"\nR2 Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# User input for prediction
sepal_length = float(input("Enter sepal length in cm: "))
sepal_width = float(input("Enter sepal width in cm: "))
petal_length = float(input("Enter petal length in cm: "))
petal_width = float(input("Enter petal width in cm: "))

# Make a prediction with feature names
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
prediction_encoded = model.predict(input_data)[0]

# Decode the predicted label
predicted_species = le.inverse_transform([prediction_encoded])[0]

# Display the predicted species
print(f"\nThe predicted species for the given input is: {predicted_species}")
