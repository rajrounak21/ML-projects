# important libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the dataset
credit_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\creditcard.csv")
print(credit_data.head())
# shape of dataset
print(credit_data.shape)
# find null values in dataset
print(credit_data.isnull().sum())
#distribution of legal transaction and fraudulent transaction
print(credit_data['Class'].value_counts())
# this dataset is  highly imbalanaced dataset
legal_transaction=credit_data[credit_data.Class==0]
fraud_transaction=credit_data[credit_data.Class==1]
print(legal_transaction.shape)
print(fraud_transaction)
# perform statical measures
print(legal_transaction.describe())

print(fraud_transaction.describe())
# comparing tha  value in both transaction
print(credit_data.groupby('Class').mean())


# build a sample dataset that containg similar distribution of normal transaction and fraud tranaction
sample_data=legal_transaction.sample(n=492)
# concating the both data

new_dataset=pd.concat([sample_data,fraud_transaction],axis=0)
print(
    new_dataset
    .head()
)

# assign X is equal to features excluding class
X=new_dataset.drop(columns='Class',axis=1)
# assign Y is equal to label only Class
Y=new_dataset['Class']
# splitting dataset into train test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
# accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuraccy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuraccy)

#accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuraccy=accuracy_score(X_test_prediction,Y_test)
print(testing_data_accuraccy)

# testing the model

input_data=(12	,-0.752417043,	0.345485415,2.057322913,	-1.468643298	,-1.15839368,	-0.077849829	,-0.608581418	,0.003603484,	-0.436166984,	0.747730827	,-0.793980603	,-0.770406729,	1.047626997	,-1.066603681,	1.106953457	,1.660113557,	-0.279265373,-0.419994141,0.432535349	,0.263450864,0.499624955,1.353650486,-0.25657328,-0.065083708,-0.039124354,-0.087086473,-0.1809975,0.129394059,15.99)
input_data_as_np_array=np.asarray(input_data)
input_data_reshped=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshped)

if(prediction[0]==1):
    print("ALERT! FRAUD")

else:
    print("Normal Transaction")