# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data-
 ```
Clean and format your data
Split your data into training and testing sets
```
2. Define your model
```
Use a sigmoid function to map inputs to outputs
Initialize weights and bias terms
```
3. Define your cost function
```
Use binary cross-entropy loss function
Penalize the model for incorrect predictions
```
4. Define your learning rate
```
Determines how quickly weights are updated during gradient descent
```
6.Train your model
```
Adjust weights and bias terms using gradient descent
Iterate until convergence or for a fixed number of iterations
```
6.Evaluate your model
```
Test performance on testing data
Use metrics such as accuracy, precision, recall, and F1 score
```
7.Tune hyperparameters
```
Experiment with different learning rates and regularization techniques
```
8.Deploy your model
```
Use trained model to make predictions on new data in a real-world application.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ROSHINI R K
RegisterNumber:  212222230123
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Initial data set:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/f8deb36a-96cc-4744-bead-d4b0ee2d7112)
### Data info:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/5cc2ae9b-6e59-4bfd-a437-1a4c64f64d46)
### Optimization of null values:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/fb1f31b6-8d68-4f9d-90e7-1628ca709441)
### Assignment of x and y values:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/93173468-5e9a-48a9-a65b-30821f8a727c)
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/d359669d-cfbd-4cfb-a2d5-4f83464d9f9c)
### Converting string literals to numerical values using label encoder:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/9280d42d-a07c-45d0-980f-0b56c89c0649)
### Accuracy:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/ccee6012-a206-4c9b-a689-4aee74302378)
### Prediction:
![image](https://github.com/SASIDEVIvenaram/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707332/d113bd74-982f-43e1-8401-b8726b0e6576)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

