import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# read the dataset
sonar_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\Copy of sonar data.csv",header=None)
print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data[60].value_counts())
# determine the features and label
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X_train.shape,X_test.shape)
model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy on taining data
X_train_predicted = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predicted,Y_train)
print(training_data_accuracy)
# accuracy on test data
X_test_predicted = model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_predicted,Y_test)
print(testing_data_accuracy)

#prediction
input_data=(0.0079,	0.0486,	0.0055,	0.025,	0.0344,	0.654	,0.0528,	0.7768,	0.1009,	0.124,	0.1097,	0.1215,	0.1874,	0.3383,	0.3227,	0.2723,	0.3943,	0.6432	,0.7271,	0.8673,	0.9674,	0.9847	,0.948,	0.8036,	0.6833,	0.5136	,0.309,	0.8532,	0.4019,	0.2344,	0.9905,	0.8735,	0.9817,	0.9851,	0.4589,	0.6549,	0.7382,	0.6589	,0.9089,	0.8989,	0.7643,	0.8039,	0.6591,	0.9019,	0.5078,	0.6063,	0.8702,	0.6092,	0.1052,	0.2966,	0.7674 ,0.9076,	0.0127,	0.0088,	0.0098,	0.0019,	0.0059,	0.0058,	0.0059,	0.0032)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)

if (prediction[0]=='R'):
    print("The Object is Rock")

else:
    print("The object is Mine")