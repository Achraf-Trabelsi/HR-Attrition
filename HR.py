
from numpy import linspace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
#Understanding the dataset
data=pd.read_csv("Human_Resources_Employee_Attrition.csv")
print(data.shape)
print(data.columns)
#print(data.isnull().sum())
print(data.describe())
#Matrice de correlation
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
x=linspace(1,14999,14999)
plt.plot(x,data.left)
plt.show()
#creation de un train/test dataset
label=data.left
#print(label)
dt=data
dt=dt.drop(labels='left',axis=1)
#print(dt.columns)
d_train,d_test,l_train,l_test=train_test_split(dt,label,test_size=0.33)
#print(d_test,d_train,l_train,l_test)
enc=OrdinalEncoder()
d_train=enc.fit_transform(d_train)
d_test=enc.fit_transform(d_test)
clf=RandomForestClassifier(max_depth=3,random_state=0)
print(clf)
clf.fit(d_train,l_train)
print(clf.predict(d_test))
print(accuracy_score(l_test,clf.predict(d_test)))