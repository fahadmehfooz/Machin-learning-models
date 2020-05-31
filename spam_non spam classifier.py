import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

df=pd.read_csv("spam.csv")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df['EmailText'],df['Label'],test_size=.2)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
cv=CountVectorizer()
features=cv.fit_transform(x_train)
model=svm.SVC()
model.fit(features,y_train)
ft=cv.fit_transform(x_test)
model.score(ft,y_test)