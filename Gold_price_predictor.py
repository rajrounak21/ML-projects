import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\gld_price_data.csv")
print(gold_data.head())
print(gold_data.shape)
print(gold_data.info())
print(gold_data.isnull().sum())
print(gold_data.describe())

# features and label
X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,Y_train)
#  #accuracy on training dataset
X_train_prediction=model.predict(X_train)
training_data_accuracy=metrics.r2_score(X_train_prediction,Y_train)
print("Accuracy on training dataset : ",training_data_accuracy)

 #accuracy on testing dataset
X_test_prediction=model.predict(X_test)
testing_data_accuracy=metrics.r2_score(X_test_prediction,Y_test)
print("Accuracy on testing dataset : ",testing_data_accuracy)


input_data=(1380.949951,72.779999,	15.834,	1.48021)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
predicton =model.predict(input_data_reshaped)
print(predicton)
