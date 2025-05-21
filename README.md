# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SURIYA PRAKASH.S
RegisterNumber: 212223100055

```

```

import numpy as np
import pandas as pd

data=pd.read_csv("/content/Salary.csv")
data.head(5)

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()

x=data[['Position','Level']]
x

y=data[['Salary']]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test, y_pred)
r2

print("SURIYA PRAKASH.S[212223100055]")
dt.predict([[5,6]])
```

## Output:

### initial data
![image](https://github.com/user-attachments/assets/c004ebd0-6d90-4dad-b3a9-2774ef82c9ea)

### info()

![image](https://github.com/user-attachments/assets/42869789-ae4b-499c-a4d3-aeca88f48e75)

### LabelEncoder
![image](https://github.com/user-attachments/assets/0e925636-0176-447f-9fcf-355eba7f409e)

### x and y data
![image](https://github.com/user-attachments/assets/42625a32-5623-48cb-be68-609e063bc35a)

![image](https://github.com/user-attachments/assets/7f346b4f-a8ae-4b44-9a41-1a23dc3d2a97)

### mean_squared_error

![image](https://github.com/user-attachments/assets/c6cf4e86-93c5-4bde-a992-fb2f29da249a)

### r2_score
![image](https://github.com/user-attachments/assets/bd6e1b9f-ad99-4256-96f9-dddbadb9f824)
### prediction
![image](https://github.com/user-attachments/assets/2ae327fa-37dc-417b-ad5b-583631da5670)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
