
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv("Human_Resources_Employee_Attrition.csv")
print(data.shape)
print(data.columns)
#print(data.isnull().sum())
print(data.describe())

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

label=data.left
#print(label)
dt=data
dt=dt.drop(labels='left',axis=1)
#print(dt.columns)
d_train,d_test,l_train,l_test=train_test_split(dt,label,test_size=0.33)
#print(d_test,d_train,l_train,l_test)
model=MultinomialNB()
model.fit(d_train,l_train)
model.predict(d_test[:])
