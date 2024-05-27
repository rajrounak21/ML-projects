import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset=pd.read_csv("C:\\Users\\rouna\\Downloads\\winequality-red.csv")
print(wine_dataset.head())
print(wine_dataset.describe())
#number of values foe each quality
#sns.catplot(x='quality',data=wine_dataset,kind='count')

#volatile acidity vs quality
# plot =plt.figure(figsize=(5,5))
# sns.barplot(x='quality',y='volatile acidity',data=wine_dataset)
# #citric acid vs Quality
# plot =plt.figure(figsize=(5,5))
# sns.barplot(x='quality',y='citric acid',data=wine_dataset)



#correlation

correlation=wine_dataset.corr()
# consructing a heatmapto understand the correlation between column
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()

# x assign features and y assign label
X=wine_dataset.drop(columns='quality',axis=1)
Y=wine_dataset['quality'].apply(lambda y_value:1 if y_value>=7 else 0)

#train test model
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=3)
model=RandomForestClassifier()
model.fit(X_train,Y_train)

# accuracy on training dataset
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on training dataset : ",training_data_accuracy)

 #accuracy on testing dataset
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing dataset : ",testing_data_accuracy)

input_data=(9.3,	0.37,	0.44,	1.6,	0.038	,21	,42	,0.99526,	3.24,	0.81,10.8)
input_data_as_np_array=np.asarray(input_data)
input_data_reshaped=input_data_as_np_array.reshape(1,-1)
predicton =model.predict(input_data_reshaped)
print(predicton)

if (predicton[0]==1):
    print("Good Quality Wine")

else:
    print("Bad Quality Wine")
