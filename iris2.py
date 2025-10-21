import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
iris_data = pd.read_csv('iris.csv')
#print(iris_data.head())  // print all the contents of the dataset
#split the data into features x and y
x = iris_data.drop(columns=['Id','Species'])
y = iris_data['Species']
#split the data into trainig and testing state
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
#standardize the features
scaler = StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#print(x.head())  // print the values of the co,rant , except species and id
#create ml model
model = LogisticRegression();
#model.fit(x.values,y) #train the model with initial values
model.fit(x_train_scaled,y_train) #train the model with standardized values
#predictions = model.predict([[2.1,40.2,5.1,3.2]]) #values given to predict 
#print(predictions)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy*100)
new_data = np.array([[5.1,3.5,1.4,0.2],[2.1,40.2,5.1,3.2]])
#standardize the new data
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("Predictions:",predictions)


import pickle
pickle.dump(model,open('model.pkl','wb'))