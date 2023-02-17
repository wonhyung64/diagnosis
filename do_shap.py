#%%
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
# %%
path = "/Users/wonhyung64/data/diagnosis"
frcnn_df = pd.read_csv(f"{path}/pascal_frcnn.csv")
retina_df = pd.read_csv(f"{path}/pascal_retina.csv")
frcnn_df = frcnn_df.drop(["pos_iou"], 1)
retina_df = retina_df.drop(["pos_iou"], 1)

frcnn_5 = frcnn_df[(frcnn_df["type"] == 5.0) | (frcnn_df["type"] == 0.0)]
frcnn_5["type"] = frcnn_5["type"].map(lambda x: 1.0 if x > 0. else x)

retina_3 = retina_df[(retina_df["type"] == 3.0) | (retina_df["type"] == 0.0)]
retina_3["type"] = retina_3["type"].map(lambda x: 1.0 if x > 0. else x)

model_linear = sklearn.linear_model.LogisticRegression(max_iter=10000)
model_linear.fit(frcnn_5.loc[:, "label":], frcnn_5["type"])

def model_linear_proba(x):
    return model_linear.predict_proba(x)[:,1]

def model_linear_log_odds(x):
    p = model_linear.predict_log_proba(x)
    return p[:,1] - p[:,0]

sample_ind = 18
fig, ax = shap.partial_dependence_plot(
    "ctr_x", model_linear_proba, frcnn_5.loc[:, "label":], model_expected_value=True,
    feature_expected_value=True, show=False, ice=False
)

#%%
model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(frcnn_5.loc[:, "label":], frcnn_5["type"])
explainer = shap.Explainer(model, frcnn_5.loc[:, "label":])
shap_values = explainer(frcnn_5.loc[:, "label":])
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)


model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(retina_3.loc[:, "label":], retina_3["type"])
explainer = shap.Explainer(model, retina_3.loc[:, "label":])
shap_values = explainer(retina_3.loc[:, "label":])
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

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