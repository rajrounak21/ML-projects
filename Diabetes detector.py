#import all libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

diabetes_data=pd.read_csv("C:\\Users\\rouna\\Downloads\\diabetes.csv")
print(diabetes_data.head())
print(diabetes_data.shape)

print(diabetes_data['Outcome'].value_counts().mean)
print(diabetes_data.groupby('Outcome').mean())

#X for features and Y for label
X=diabetes_data.drop(columns='Outcome',axis=1)
Y=diabetes_data['Outcome']
print(X)
print(Y)

#Data Standardization

standard=StandardScaler()

standardized_data=standard.fit_transform(X)
print(standardized_data)
#features and label
X=standardized_data
Y=diabetes_data['Outcome']

# split the data in test train and train the model
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=SVC(kernel='linear')
model.fit(X_train,Y_train)

#Accuracy score for training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of the training data : ",training_data_accuracy)
# Acuuracy score for testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of the training data : ",testing_data_accuracy)



# predict the patient is diabetic or not diabetic
input_data=(5,139,64,35,140,77.6,0.411,26)
# changing the input data as array
input_data_as_np_array=np.asarray(input_data)
# reshape the dataset
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
# standardization input data
std_data = standard.transform(input_data_reshaped)
prediction=model.predict(std_data)
print(prediction)


if (prediction[0]==0):
    print("The person is non diabetic")

else:
    print("The person is diabetic")