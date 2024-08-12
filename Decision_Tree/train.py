from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from Decision_Tree import Decision_Tree
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
data=load_breast_cancer()
X,y=data.data,data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

clf=Decision_Tree()
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

print(accuracy_score(y_test,predictions))

clf2=RandomForest()
clf2.fit(X_train,y_train)
predictions=clf2.predict(X_test)
print(accuracy_score(y_test,predictions))

#random forest has  a better accuracy score than decision trees