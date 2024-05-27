import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
big_mart_sales_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\Train.csv")
print(big_mart_sales_data.head())
print(big_mart_sales_data.keys())
print(big_mart_sales_data.shape)
# find the null values
print(big_mart_sales_data.isnull().sum())
# fill the null value in item_weight with mean value
big_mart_sales_data['Item_Weight'].mean()
big_mart_sales_data['Item_Weight'].fillna(big_mart_sales_data['Item_Weight'].mean(),inplace=True)
# now fill the outlet size missing value with mode

big_mart_sales_data['Outlet_Size'].mode()
Mode_of_Outlet_Size=big_mart_sales_data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))
print(Mode_of_Outlet_Size)
miss_value=big_mart_sales_data['Outlet_Size'].isnull()
print(miss_value)
big_mart_sales_data.loc[miss_value,'Outlet_Size']=big_mart_sales_data.loc[miss_value,'Outlet_Type'].apply(lambda x:Mode_of_Outlet_Size[x])
print(big_mart_sales_data.isnull().sum())
#in Item_Fat_Content there are shortcuts are present
big_mart_sales_data.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)
print(big_mart_sales_data['Item_Fat_Content'].value_counts())

encoder=LabelEncoder()
big_mart_sales_data['Item_Identifier'] = encoder.fit_transform(big_mart_sales_data['Item_Identifier'])

big_mart_sales_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_sales_data['Item_Fat_Content'])

big_mart_sales_data['Item_Type'] = encoder.fit_transform(big_mart_sales_data['Item_Type'])

big_mart_sales_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_sales_data['Outlet_Identifier'])

big_mart_sales_data['Outlet_Size'] = encoder.fit_transform(big_mart_sales_data['Outlet_Size'])

big_mart_sales_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_sales_data['Outlet_Location_Type'])

big_mart_sales_data['Outlet_Type'] = encoder.fit_transform(big_mart_sales_data['Outlet_Type'])
#x contain features  and Y contain label
X=big_mart_sales_data.drop(columns='Item_Outlet_Sales',axis=1)
Y=big_mart_sales_data['Item_Outlet_Sales']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(X_train,Y_train)

# R squarred error on training data
X_train_prediction=model.predict(X_train)
r2_score_on_training_data=r2_score(X_train_prediction,Y_train)
print("R Squarred Value :",r2_score_on_training_data)
# R squarred error on testing data
X_test_prediction=model.predict(X_test)
r2_score_on_testing_data=r2_score(X_test_prediction,Y_test)
print("R Squarred Value :",r2_score_on_testing_data)
input_data=(211,	13,	13.8,	0,	0.058091482,	4,	245.1802,	6,	2004,	2,	1,	1)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)