# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:27:57 2021

@author: m-lin
"""

# ライブラリの読み込み
import pandas as pd
from pandas_profiling import ProfileReport

# データの読み込み
train = pd.read_csv('./data/train.tsv', sep='\t')

# idの削除
train.drop('id', axis = 1, inplace=True)

# pandas-profiling
profile = ProfileReport(train)
profile.to_file('profile_report.html')

# クロス集計
pd.crosstab(train['Y'], train['odor'], margins=True)
pd.crosstab(train['Y'], train['gill-color'], margins=True)

'''
仮説検証
'''

# ライブラリの読み込み
from sklearn.metrics import accuracy_score

# trainデータのYを予測
y_train_preds = []

for i in range(len(train)):
    
    # 'odor'がa,l,nのとき'Y'はe
    if train['odor'][i] == 'a':
        y_train_preds.append('e')
        
    elif train['odor'][i] == 'l':
        y_train_preds.append('e')
        
    elif train['odor'][i] == 'n':
        y_train_preds.append('e')
        
    # それ以外は'Y'はp
    else:
        y_train_preds.append('p')

# accuracyを計算する
score = accuracy_score(train['Y'], y_train_preds)
print(score)