# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Step 2: Load the dataset
iris_data = pd.read_csv('iris.csv')

# Step 3: Split features (x) and target (y)
x = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Step 4: Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 5: Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(x_train_scaled, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained successfully with accuracy: {accuracy*100:.2f}%")

with open("accuracy.txt", "w") as f:
    f.write(f"{accuracy*100:.2f}")
# Step 8: Save the model and the scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ model.pkl and scaler.pkl created successfully!")
