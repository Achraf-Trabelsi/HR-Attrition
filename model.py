import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
import pickle

# Understanding the dataset

data = pd.read_csv("Dataset/Human_Resources_Employee_Attrition.csv")
print(data.shape)
print(data.columns)
# print(data.isnull().sum())
print(data.describe())

# Data visualization

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# creation de un train/test dataset

label = data.left
# print(label)
dt = data
dt = dt.drop(labels="left", axis=1)
# print(dt.columns)
d_train, d_test, l_train, l_test = train_test_split(dt, label, test_size=0.33)

# data preprocessiong

enc = OrdinalEncoder()
d_train = enc.fit_transform(d_train)
d_test = enc.fit_transform(d_test)

# model selection

clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(d_train, l_train)
print(clf.predict(d_test))

# accurracy_score
print("accuracy is : ", accuracy_score(l_test, clf.predict(d_test)))

# putting the model into a file
with open("weights/HRattrition_model_RandomForest.pkl", "wb") as file:
    pickle.dump(clf, file)
