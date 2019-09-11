import pandas as pd 
import numpy as np
import sys
import scipy
import IPython
import random 
import time 
import warnings
from subprocess import check_output

## model algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier#from xgboost import XGBClassfier

## model helper
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

## visualization

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix




warnings.filterwarnings('ignore')
print('-'*25)


df_train = pd.read_csv("titanic/train.csv")

print df_train.describe()

print df_train.describe(include=['O'])
df_test = pd.read_csv("titanic/test.csv")

combine = [df_train,df_test]
### explore 

#print df_train.describe()
#print df_train.describe(include=['O'])

#sib = df_train[['SibSp' ,'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
#print sib 

#parch = df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#print parch

#grid = sns.FacetGrid(df_train, row='Embarked')#, size=2.2, aspect=1.6)
#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
#grid.add_legend()
#plt.show()

### preprocess

## cleaning 

for dataset in combine:
	dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
	dataset["FamilySize"] = dataset['SibSp'] + dataset["Parch"] + 1
	dataset["Sex"] = dataset['Sex'].map({'female': 1,'male':0 }).astype(int)
	dataset["Embarked"] = dataset['Embarked'].map({'S': 0,'C':1, 'Q':'2' }).astype(int)
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[ dataset['Age'] > 64, 'Age']
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)
	dataset['Age'] = dataset['Age'].astype(int)
drop_col = ['Ticket', 'Name', "Cabin", "PassengerId"]

df_train.drop(drop_col, axis = 1, inplace = True)
df_test.drop(['Ticket', 'Name', "Cabin"], axis = 1, inplace = True)

#print df_train.shape, df_test.shape#

#print df_train.isnull().sum(), df_test.isnull().sum()#

#print df_train.head()

#print df_train.head()
X_train = df_train.drop('Survived', axis = 1)
Y_train = df_train['Survived']
X_test = df_test.drop('PassengerId', axis =1).copy()



