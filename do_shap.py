#%%
import re
import os
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def visualize_shap(shap_values, col_num, filename):
    nrows = col_num // 2 + col_num % 2
    axes_idx = [[r,c] for r in range(nrows) for c in range(2)]

    fig_local, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(20,10))
    for i in range(col_num):
        r, c = axes_idx[i]
        shap.plots.scatter(shap_values[:, i], ax=axes[r, c], show=False)
    fig_local.suptitle(f"Type {j} Local Shap Values\n({filename.split('.')[0]})", fontsize=15)
    fig_local.tight_layout()

    fig_global = plt.figure(figsize=(30,10))
    plt.subplot(2,1,1)
    shap.plots.beeswarm(shap_values, show=False)
    plt.subplot(2,1,2)
    shap.plots.bar(shap_values, show=False)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig_global.suptitle(f"Type {j} Global Shap Values\n({filename.split('.')[0]})", fontsize=15)
    fig_global.tight_layout()

    return fig_local, fig_global


# %%
type_dict = {
    1: "label",
    2: "area",
    3: "fg_bg_ratio"
}
ex_files = [filename for filename in os.listdir() if filename.__contains__(".csv")]
for i in tqdm(range(len(ex_files))):
    filename = ex_files[i]
    df = pd.read_csv(f"{filename}")
    print(f"{filename} type summary")
    print(df["type"].value_counts())
    # for j in range(1,4):
    #     df_type = df[(df["type"] == 0) | (df["type"] == j)]
    #     X = df_type.loc[:, "label":]
    #     X = X.drop(columns=type_dict[j])
    #     y = df_type.loc[:, "type"].map(lambda x: 1. if x == j else x)

    #     model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)
    #     explainer = shap.Explainer(model, X)
    #     shap_values = explainer(X)

    #     fig_local, fig_global = visualize_shap(shap_values, col_num=len(X.columns), filename=filename)

    #     fig_local.savefig(f"./ex3/{filename.split('.')[0]}_type_{j}_local.png")
    #     fig_global.savefig(f"./ex3/{filename.split('.')[0]}_type_{j}_global.png")

    X = df.loc[:, "label":]
    y = df.loc[:, "type"].map(lambda x: 1. if x >= 1 else x)

    model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    fig_local, fig_global = visualize_shap(shap_values, col_num=len(X.columns), filename=filename)

    fig_local.savefig(f"./ex3/{filename.split('.')[0]}_type_{j}_local.png")
    fig_global.savefig(f"./ex3/{filename.split('.')[0]}_type_{j}_global.png")


#%%
shap_values.data.shape
shap_values.base_values.shape
X
y
df.type.value_counts()
i=3
for i in range(8):
    print(
        np.mean(shap_values.values[:, i][shap_values.values[:, i] > 0.])
    )
    filename
shap_values.data[:, 0] * \
shap_values.values[:, 0]

shap_values.data[:, 1] * \
shap_values.values[:, 1]
X
#%%
shap.summary_plot(shap_values)
shap.plots.waterfall(shap_values[924])
shap.plots

model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(retina_3.loc[:, "label":], retina_3["type"])
explainer = shap.Explainer(model, retina_3.loc[:, "label":])
shap_values = explainer(retina_3.loc[:, "label":], check_additivity=False)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
shap.plots.bar(shap_values.mean(0), ax=axes[0])

shap.plots.bar(shap_values.max(0))
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values.max(0))
shap.plots.scatter(shap_values[:, "label"], color=shap_values)
shap.plots.waterfall(shap_values[0])


import tensorflow as tf
from tqdm import tqdm
anchor_boxes = AnchorBox().get_anchors(512, 512)
anchor_boxes = tf.cast(anchor_boxes, dtype=tf.float64)
iou_pos_list = []
num_pos_list = []
num_neg_list = []

for i in tqdm(range(len(retina_df))):

    y1, x1, y2, x2, label = retina_df.loc[i, "box_y1": "label"]
    gt_box = tf.constant([[y1, x1, y2, x2]], dtype=tf.float64)
    iou_matrix = compute_iou(anchor_boxes, gt_box * 512)
    max_iou = tf.reduce_max(iou_matrix, axis=1)
    matched_gt_idx = tf.argmax(iou_matrix, axis=1)
    positive_mask = tf.greater_equal(max_iou, 0.5)
    pos_iou = max_iou[positive_mask]
    pos_num = tf.reduce_sum(tf.cast(positive_mask, dtype=tf.int32))
    negative_mask = tf.less(max_iou, 0.4)
    neg_num = tf.reduce_sum(tf.cast(negative_mask, dtype=tf.int32))

    iou_pos_list.append(pos_iou.numpy().tolist())
    num_pos_list.append(pos_num.numpy())
    num_neg_list.append(neg_num.numpy())
    
retina_df["pos_num2"] = num_pos_list

import matplotlib.pyplot as plt
plt.hist(retina_df["pos_num2"])
plt.hist(retina_df["pos_num"])



#%%
frcnn_5

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


retina_3
X = retina_3.loc[:, "label":]
y = retina_3.loc[:, "type"]
retina_3
X = frcnn_5.loc[:, "label":]
y = frcnn_5.loc[:, "type"]

onehot_encoder = OneHotEncoder()
label_onehot = onehot_encoder.fit_transform(np.expand_dims(X["label"].to_numpy(), -1)).toarray()
X = X.drop("label", axis=1)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = np.concatenate([label_onehot, X], 1)

model = xgboost.XGBClassifier(n_estimators=100, max_depth=2)
model.fit(X, y)
model.feature_importances_
frcnn_df["type"].value_counts()
retina_df["type"].value_counts()
