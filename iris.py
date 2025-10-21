import pandas as pd
from sklearn.linear_model import LogisticRegression
iris_data = pd.read_csv('iris.csv')
#print(iris_data.head())  // print all the contents of the dataset
#split the data into features x and y
x = iris_data.drop(columns=['Id','Species'])
y = iris_data['Species']
#print(x.head())  // print the values of the content , except species and id
#create ml model
model = LogisticRegression();
model.fit(x.values,y) #train the model
predictions = model.predict([[2.1,40.2,5.1,3.2]]) #values given to predict 
print(predictions)