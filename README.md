# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages. 
2. Read the given csv file and display the few contents of the data. 
3. Assign the features for x and y respectively. 
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NAVEEN S
RegisterNumber: 212222110030 
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```
## Output:
## 1. Result output
![243066515-c08810ba-4b33-43e0-86b9-88e149733769](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/7955deac-d453-4189-b660-1e82563899d1)
## 2. data.head()
![243066532-06725080-0126-4ff8-8a80-bb89d4c3addc](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/aef1dd5f-18b7-4996-b0e6-f4315b816773)
## 3. data.info()
![243066617-0bdf2600-0403-4357-a19b-411728f33895](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/15aa2ac6-d548-4c28-ac04-46d398a1b53c)
## 4. data.isnull().sum() 
![243066622-9088861a-060c-4063-9a9b-aac56267da47](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/af88de0c-0d8d-4b54-a832-679765891051)
## 5. Y_prediction value
![243066650-f05da199-a5b1-49a4-ab2c-e431e481082d](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/c14755a4-d305-478d-bdce-603c74c271bb)
## 6. Accuracy value
![243066667-ba64d866-04f4-48d6-8d5b-fb253524473a](https://github.com/NaveenSivamalai/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123792574/f0744410-4410-4110-826e-4d12218495dd)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
