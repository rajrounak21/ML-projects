import pandas as  pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
insurance_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\insurance.csv")
print(insurance_data.head())
# sns.set()
# plt.figure(figsize=(6,6))
# sns.histplot(insurance_data['age'])
# plt.title('Age Distribution')
# plt.show()

# replace the data in sex smoker region
print(insurance_data['region'].value_counts())
print(insurance_data['sex'].value_counts())
print(insurance_data['smoker'].value_counts())
insurance_data.replace({'sex':{'male':0 , 'female':1}},inplace=True)
insurance_data.replace({'smoker':{'yes':0 , 'no':1}},inplace=True)
insurance_data.replace({'region':{'southeast':0 , 'southwest':1 , 'northwest':2, 'northeast':3}},inplace=True)

print(insurance_data.head())
#X is features and Y is label
X=insurance_data.drop(columns='charges',axis=1)
print(X.head())
Y=insurance_data['charges']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)

#calculate r2 value on training data
X_train_prediction=model.predict(X_train)
r2_value=r2_score(X_train_prediction,Y_train)
print("R squared value :",r2_value)


#calculate r2 value on training data
X_test_prediction=model.predict(X_test)
r2_value=r2_score(X_test_prediction,Y_test)
print("R squared value :",r2_value)


#testing the model=
input_data=(32 ,   0,  28.880,         0,       1,       2)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)