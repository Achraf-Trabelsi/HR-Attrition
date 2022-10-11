import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Understanding the dataset

data = pd.read_csv("Dataset/Human_Resources_Employee_Attrition.csv")
# print(data.isnull().sum())

# Data Visualization

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Creation de un train/test dataset

label = data.left
# print(label)
dt = data
dt = dt.drop(labels="left", axis=1)
# print(dt.columns)
d_train, d_test, l_train, l_test = train_test_split(dt, label, test_size=0.33)
d_test.to_csv("test.csv", index=False)

# Data Preprocessiong

enc = OrdinalEncoder()
d_train = enc.fit_transform(d_train)
d_test = enc.fit_transform(d_test)

# Random Forest

clf_1 = RandomForestClassifier(max_depth=4, random_state=42)
clf_1.fit(d_train, l_train)
print(clf_1.predict(d_test))
acc_1 = accuracy_score(l_test, clf_1.predict(d_test))
fscore_1 = f1_score(l_test, clf_1.predict(d_test))
print("accuracy is : ", acc_1)
print("f1 score is : ", fscore_1)

# Gradient Boost

clf_2 = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.01, max_depth=4, random_state=42
)
clf_2.fit(d_train, l_train)
print(clf_2.predict(d_test))
acc_2 = accuracy_score(l_test, clf_2.predict(d_test))
fscore_2 = f1_score(l_test, clf_2.predict(d_test))
print("accuracy is : ", acc_2)
print("F1 score is : ", fscore_2)
# Putting the model into a file

with open("weights/best_model.pkl", "wb") as file:
    if acc_2 < acc_1:
        pickle.dump(clf_1, file)
    else:
        pickle.dump(clf_2, file)
