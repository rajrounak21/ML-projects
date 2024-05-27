import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
heart_disease_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\heart_disease_data.csv")
print(heart_disease_data.head())
print(heart_disease_data.isnull().sum())
print(heart_disease_data.shape)
print(heart_disease_data['target'].value_counts())
# 1-->> Defective heart
# 0-->> Healthy heart
X=heart_disease_data.drop(columns='target',axis=1)
Y=heart_disease_data['target']
#split the data in train test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
# fit the dataset into logistic regression
model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training data : ",training_data_accuracy)


#accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy= accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing data : ",testing_data_accuracy)

#Testing the model

input_data=(38,	0,	0,	178,	228,	1,	1,	165,	1,	1,	1,	2,	3)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)


if(prediction[0]==0):
    print("The patient have Healthy heart")

else:
    print("The patient have Defective heart")