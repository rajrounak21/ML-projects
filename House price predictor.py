import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
boston_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\boston.csv")
# view some rows and column
print(boston_data.head())
# shape of dataframe
print(boston_data.shape)
# find missing value
print(boston_data.isnull().sum())
#statics calculation
print(boston_data.describe())

correlation=boston_data.corr()
print(correlation)

# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
# plt.show()
# train data
X=boston_data.drop(columns='MEDV',axis=1)
# test data
Y=boston_data['MEDV']
print(X)
print(Y)
# training and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

model = LinearRegression()
model.fit(X_train,Y_train)

#  accuracy on training data prediction
training_data_prediction=model.predict(X_train)
score_1= r2_score (training_data_prediction,Y_train)
print(score_1)

#accuracy on  testing data prediction
testing_data_prediction=model.predict(X_test)
score_2= r2_score (testing_data_prediction,Y_test)
print(score_2)


# predict the house price
input_data=(1.19294,	0,	21.89,	0,	0.624,	6.326,	97.7,	2.271,	4,	437	,21.2,	396.9,	12.26)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)