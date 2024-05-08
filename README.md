# Ex-No:4 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data. 
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy, confusion matrices.
5. Display the results.

## Program:
```python

# Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: Krithick Vivekananda
# RegisterNumber:  212223240075


import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Dataset:
![dat](318485814-c766aca1-20e5-494c-9bd2-bf9d613d73c0.png)
### Dataset after dropping the salary column:
![alt text](318486097-e5d4d549-564a-477b-9995-0dcf529b2855.png)
### Checking if Null values are present:
![alt text](318489566-363ab4b1-1a13-4a80-9018-bf9a629638fb.png)
### Checking if duplicate values are present:
![alt text](318486828-397fbb5d-12f8-44c1-98c6-cdf834b35a15.png)
### Dataset after encoding:
![alt text](318487373-860d4c34-2690-44c1-bedf-6e0fab7100f5.png)
### X-Values:
![alt text](318487616-18e69f47-5962-45fa-8fd0-6b493daf3de7.png)
### Y-Values:
![alt text](318486904-640cf836-9b8f-422a-928b-392bc19e156e.png)
### Y_pred Values:
![alt text](318487826-bf69785e-33d1-4401-9201-948d5bfbef84.png)
### Accuracy:
![alt text](318487980-48ee4758-73e1-4808-b32b-325f98ad9100.png)
### Confusion Matrix:
![alt text](318488168-7441087f-341f-400e-b6ec-6098b1501327.png)
### Classification Report:
![alt text](318488324-20fdb76e-6cf3-45ec-b3a0-14bc1edfb606.png)
### lr.predict:
![alt text](318488572-554e98ef-e368-4c7e-94ae-198cb6192d47.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
