#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from itertools import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
#%%
path = "/Users/wonhyung64/data/diagnosis"

df = pd.read_csv(f"{path}/pascal_frcnn.csv")
df.loc[:, "pos_iou_mean"] = df["pos_iou"].map(lambda x: np.mean(eval(x)))

nodes = ["label", "area", "ratio", "ctr_x", "ctr_y", "pos_num", "pos_iou_mean"]
DAGs = []
for i in range(1, 8):
    DAGs += list(combinations(nodes, i))

score_dag = {}
df_5 = df[(df["type"] == 5) | (df["type"] == 0)]
df_5.loc[:, "type"] = df_5["type"].map(lambda x: 1 if x > 0 else x)
for col in DAGs:
    X = df_5[list(col)]
    y = df_5["type"]

    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    if "label" in X.columns:
        onehot_encoder = OneHotEncoder()
        label_onehot = onehot_encoder.fit_transform(np.expand_dims(X["label"].to_numpy(), -1)).toarray()
        X = X.drop("label", axis=1)

        if X.shape[1] == 0:
            X = label_onehot
        else: 
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X = np.concatenate([label_onehot, X], 1)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)


    # model = LogisticRegression()
    model = LogisticRegression(penalty="l1", C=0.1, solver="saga")
    model.fit(X, y)
    score = log_loss(y_true=y, y_pred=model.predict_proba(X))
    score_dag[score] = col
    np.round(model.coef_[0],3)

col = score_dag[min(score_dag.keys())]
print(f"{score_dag[min(score_dag.keys())]}: {min(score_dag.keys())}")

#%%
score_dag = {}
df_5 = df[(df["type"] == 5) | (df["type"] == 0)]
df_5.loc[:, "type"] = df_5["type"].map(lambda x: 1 if x > 0 else x)
for col in DAGs:
    X = df_5[list(col)]
    y = df_5["type"]

    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    if "label" in X.columns:
        onehot_encoder = OneHotEncoder()
        label_onehot = onehot_encoder.fit_transform(np.expand_dims(X["label"].to_numpy(), -1)).toarray()
        X = X.drop("label", axis=1)

        if X.shape[1] == 0:
            X = label_onehot
        else: 
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X = np.concatenate([label_onehot, X], 1)
    else:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    model = RandomForestClassifier()
    # model = LogisticRegression(penalty="l1", solver="saga")
    model.fit(X, y)
    score = log_loss(y_true=y, y_pred=model.predict_proba(X))
    score_dag[score] = col

col = score_dag[min(score_dag.keys())]
print(f"{score_dag[min(score_dag.keys())]}: {min(score_dag.keys())}")

np.round(model.feature_importances_, 3)

#%%
print(f"{score_dag_linear[min(score_dag_linear.keys())]}: {min(score_dag_linear.keys())}")
print(f"{score_dag_linear_scale[min(score_dag_linear_scale.keys())]}: {min(score_dag_linear_scale.keys())}")
print(f"{score_dag_penalty[min(score_dag_penalty.keys())]}: {min(score_dag_penalty.keys())}")
print(f"{score_dag_penalty_scale[min(score_dag_penalty_scale.keys())]}: {min(score_dag_penalty_scale.keys())}")

print(f"{score_dag_rf[min(score_dag_rf.keys())]}: {min(score_dag_rf.keys())}")
print(f"{score_dag_rf_scale[min(score_dag_rf_scale.keys())]}: {min(score_dag_rf_scale.keys())}")


col = ['label', 'area', 'ratio', 'ctr_x', 'ctr_y', 'pos_num', 'pos_iou_mean']
col = ('label', 'area', 'ratio', 'pos_num')
model = RandomForestClassifier()
model.fit(X, y)
model.feature_importances_
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
sns.heatmap(
    df[["type", "label", "area", "ratio", "ctr_x", "ctr_y", "pos_num", "pos_iou_mean"]].corr(),
    annot=True,
    cmap="Blues", ax=ax
    )
fig.savefig("/Users/wonhyung64/Downloads/corr.png")