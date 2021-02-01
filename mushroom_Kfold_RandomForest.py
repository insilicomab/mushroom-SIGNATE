# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:27:45 2021

@author: m-lin
"""

'''
データの読み込み
'''

# ライブラリの読み込み
import pandas as pd
import numpy as np

# データの読み込み
train = pd.read_csv('data/train.tsv', sep='\t')
test = pd.read_csv('data/test.tsv', sep='\t')

# データの確認
train.head()
test.head()
train.dtypes
test.dtypes

# データの結合
df = pd.concat([train, test], sort=False) # concat()関数で縦（行）方向に結合
df.head()
df.dtypes

# 欠損値の確認
df.isnull().sum()

'''
特徴量エンジニアリング
'''

# ライブラリのインポート
from sklearn.preprocessing import LabelEncoder

# データフレームのクリーニング
delete_columns = ['id']
df.drop(delete_columns, axis=1, inplace=True)

# trainとtestに再分割
train = df[:len(train)] 
test = df[len(train):]

'''
label Encodeing
'''

# trainのLabel Encodingによるダミー変数化
train_categories = train.columns[df.dtypes == 'object'] # カテゴリ変数の格納
print(train_categories)

for cat in train_categories:
    le = LabelEncoder()
    if train[cat].dtypes == 'object':
        le = le.fit(train[cat])
        train[cat] = le.transform(train[cat])

train.head()

# testのLabel Encodingによるダミー変数化
test['Y'].fillna('missing', inplace=True) # Y の欠損値をmissingで置き換える
test.tail()

# Label Encodingによるダミー変数化
test_categories = test.columns[test.dtypes == 'object'] # カテゴリ変数の格納
print(test_categories)

for cat in test_categories:
    le = LabelEncoder()
    if test[cat].dtypes == 'object':
        le = le.fit(test[cat])
        test[cat] = le.transform(test[cat])

test.head()

# 説明変数と目的変数に分割
Y_train = train['Y']
X_train = train.drop('Y', axis=1)
X_test = test.drop('Y', axis=1)

# 分割後のデータの確認
Y_train.head()
X_train.head()
X_test.head()

'''
モデリング（LogisticRegression）
'''

# ライブラリの読み込み
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 3分割する
folds = 3
kf = KFold(n_splits=folds)

# 機械学習モデルの学習
models = []

for train_index, val_index in kf.split(X_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    # ロジスティック回帰モデルの作成
    model = LogisticRegression(penalty='l2', solver='sag', random_state=0)
    model.fit(x_train, y_train)
    
    # 検証データにおける予測
    y_pred = model.predict(x_valid)
    print(accuracy_score(y_valid, np.round(y_pred)))
    
    models.append(model)

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)

# 予測結果の平均
preds_array = np.array(preds)
y_pred = np.mean(preds_array, axis=0)

# 0, 1変換
y_pred = (y_pred > 0.5).astype(int)

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('data/sample_submit.csv', header=None)
sub.head()

# [Y]の値を置き換え
sub[1] = y_pred
sub[1].replace([0, 1], ['e', 'p'], inplace=True) # 0:e, 1:p

# CSVファイルの出力
sub.to_csv('submission/submission_Kfold_LR.csv', header=None, index = None)