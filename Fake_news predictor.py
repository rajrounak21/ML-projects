import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
news_dataset=pd.read_csv("C:\\Users\\rouna\\Downloads\\train.csv")

print(news_dataset.head())
print(news_dataset.keys())
# find null values
print(news_dataset.isnull().sum())
# fill the null values with ' '
news_dataset=news_dataset.fillna('')
#create a new column which consist a combination of title and author
news_dataset['content']= news_dataset['author']+'  '+news_dataset['title']


#news_dataset['content'].apply(stem)
print(news_dataset['content'])
# stemming  remove words like (is ,am ,are, yourself, your, our )and etc.
ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)
news_dataset['content']=news_dataset['content'].apply(stem)
#X variable contain features and Y variable contain label
X=news_dataset['content'].values
print(X)
Y=news_dataset['label'].values
# convert the word into vector
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X=vectorizer.transform(X)
print(X)

# train test split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)

#accuracy score on the training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)
#accuracy score on the testing data
X_test_prediction=model.predict(X_test)

testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(testing_data_accuracy)


#testing the data
input_data= ('charlie mcdermid pope francis, trump, india: your tuesday brief - the new york time')
input_data_as_np_array = vectorizer.transform([input_data])
input_data_reshaped = input_data_as_np_array.toarray()

predicton = model.predict(input_data_reshaped)
print(predicton)

#print(news_dataset['content'][331])
if (predicton[0]==0):
    print("The News is Real")

else:
    print("The News is Fake")
# from joblib import dump,load
# dump(model,"Fak_data.joblib")