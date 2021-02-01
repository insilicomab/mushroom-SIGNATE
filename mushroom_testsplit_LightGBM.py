# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:30:10 2021

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
モデリング（LightGBM）
'''

# ライブラリのインポート
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# カテゴリカル変数の指定
categorical_features = list(X_train.columns) # 説明変数の指定

# 学習用・検証用データの分割
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, 
                                                    test_size = 0.3,
                                                    random_state = 0,
                                                    stratify = Y_train)

lgb_train = lgb.Dataset(x_train, y_train,
                                         categorical_feature=categorical_features)
lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train,
                                         categorical_feature=categorical_features)

# ハイパーパラメータの設定
params = {
    'objective': 'binary',
    'metric': 'binary_error'
}

# 機械学習モデルの学習
model = lgb.train(params, lgb_train,
                               valid_sets=[lgb_train, lgb_eval],
                               verbose_eval=10,
                               num_boost_round=1000, # 学習回数の実行回数
                               early_stopping_rounds=10) # early_stoppingの判定基準

# テストデータにおける予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.4).astype(int)

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
sub.to_csv('submission/submission_testsplit_LightGBM.csv', header=None, index = None)

'''
特徴量重要度の算出
'''

# ライブラリのインポート
import matplotlib.pyplot as plt

# importanceを表示する
cols = list(X_train.columns)         # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(model.feature_importance()) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
print(df_importance)

# 特徴量重要度を棒グラフでプロットする関数 
def plot_feature_importance(df): 
    n_features = len(df)                              # 特徴量数(説明変数の個数) 
    df_plot = df.sort_values('importance')            # df_importanceをプロット用に特徴量重要度を昇順ソート 
    f_importance_plot = df_plot['importance'].values  # 特徴量重要度の取得 
    plt.barh(range(n_features), f_importance_plot, align='center') 
    cols_plot = df_plot['feature'].values             # 特徴量の取得 
    plt.yticks(np.arange(n_features), cols_plot)      # x軸,y軸の値の設定
    plt.xlabel('Feature importance')                  # x軸のタイトル
    plt.ylabel('Feature')                             # y軸のタイトル

# 特徴量重要度の可視化
plot_feature_importance(df_importance)