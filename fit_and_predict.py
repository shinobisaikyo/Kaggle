import numpy as np
import pandas as pd

from load_data import load_train_data,load_test_data
from sklearn.linear_model import LogisticRegression

df = load_train_data()
x_train = df.drop(['Survived', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
y_train = df['Survived'].values

import sklearn.preprocessing as sp

le = sp.LabelEncoder()
le.fit(x_train.Sex.unique())
x_train.Sex = le.fit_transform(x_train.Sex)
one = sp.OneHotEncoder()
enced = one.fit_transform(x_train.Sex.values.reshape(1, -1).transpose())
# enced = one.fit_transform(x_train.Sex.values)
temp = pd.DataFrame(index=df.Sex.index, columns='Sex-' + le.classes_, data=enced.toarray())
enced_data = pd.concat([x_train, temp], axis=1)
del enced_data['Sex']
enced_data

# 欠損値把握
# df.isnull().any(axis=0)
# df.isnull().sum()
from sklearn.preprocessing import Imputer
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
enced_data = im.fit_transform(enced_data)

clf = LogisticRegression(random_state=0)
clf.fit(enced_data, y_train)

df_test = load_test_data()
x_test = df_test.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
x_test.Sex = le.fit_transform(x_test.Sex)
enced_test = one.fit_transform(x_test.Sex.values.reshape(1, -1).transpose())
temp_test = pd.DataFrame(index=df_test.Sex.index, columns='Sex-' + le.classes_, data=enced_test.toarray())
enced_data_test = pd.concat([x_test, temp_test], axis=1)
del enced_data_test['Sex']
enced_data_test
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
enced_data_test = im.fit_transform(enced_data_test)
pred_test = clf.predict(enced_data_test)

df_result = pd.read_csv('./gender_submission.csv')
act = df_result['Survived']
ac = np.sum(pred_test == act ) / len(pred_test)
print(ac)
