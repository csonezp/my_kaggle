import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].to_numpy()


def pipeline(data):
    num_attribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    cat_attribs = ['Sex', 'Embarked']

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    X_train: np.ndarray = full_pipeline.fit_transform(data)

    return X_train


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')


traindata = pd.read_csv('mldata/train.csv')

testdata = pd.read_csv('mldata/test.csv')

all_data = pd.concat([traindata, testdata], ignore_index=True)

all_data.Embarked[all_data.Embarked.isnull()] = all_data.Embarked.dropna().mode().values

all_data['Cabin'] = all_data.Cabin.fillna('U0')
# create feature for the alphabetical part of the cabin number
all_data['CabinLetter'] = all_data['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
# convert the distinct cabin letters with incremental integer values
all_data['CabinLetter'] = pd.factorize(all_data['CabinLetter'])[0]






print(traindata.info())

all_feature = all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# 用pipeline处理训练集
X_all: np.ndarray = pipeline(all_feature)

X_train = X_all[:891]

# test data
X_test = X_all[891:]

# train label
y_train = traindata['Survived']

# 随机森林
rf = RandomForestClassifier(n_estimators=32, max_features=10)  # 这里使用了默认的参数设置
rf.fit(X_train, y_train)  # 进行模型的训练

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

param_grid = [
    {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], 'gamma': [1, 0.1, 0.01, 0.001],
     'eval_metric': [['logloss', 'auc', 'error']]},
]
grid_search = GridSearchCV(xgb, param_grid, cv=10, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_, grid_search.best_score_)

# pred = rf.predict(X_train)

# train_score = accuracy_score(y_train, pred)
# print(train_score)

pred = grid_search.predict(X_test)

testdata['Survived'] = pred

# conf_mx = confusion_matrix(y_train, pred)

outdata = testdata[['PassengerId', 'Survived']]
outdata.to_csv("zp_titanic_6.csv", index=False, header=True)

exit()
