#%%
import os
import json
import numpy as np
import pandas as pd
# import jax.numpy as jnp
import tensorflow as tf
# from jax import grad, jit, vmap
# from jax import random
# from jax import device_put
# %%
# key = random.PRNGKey(0)
# x = random.normal(key, (10,))
# print(x)

# size = 3000
# x = random.normal(key, (size, size), dtype=jnp.float32)
# # %timeit jnp.dot(x, x.T).block_until_ready()

# x = np.random.normal(size=(size, size)).astype(np.float32)
# # %timeit jnp.dot(x, x.T).block_until_ready()


# x = np.random.normal(size=(size, size)).astype(np.float32)
# x = device_put(x)
# %timeit jnp.dot(x, x.T).block_until_ready()

#%%

def find_fn(gt_boxes, gt_labels, final_bboxes, final_labels, total_labels, iou_thresh=0.5):
    tp_boxes = []
    tp_labels = []
    fn_boxes = []
    fn_labels = []
    for c in range(total_labels):
        # continue if no gt for c
        if not tf.math.reduce_any(gt_labels == c):
            continue

        final_bbox = final_bboxes[final_labels == c]
        gt_box = gt_boxes[gt_labels == c]
        gt_label = gt_labels[gt_labels == c]
        
        # continue if no pred for c gt
        if tf.shape(final_bbox)[0] == 0:
            fn_boxes.append(gt_box)
            fn_labels.append(gt_label)
            continue

        # if there are gt and final
        iou = compute_iou(final_bbox, gt_box)
        loc_match = iou >= iou_thresh
        fn_bool = tf.logical_not(tf.reduce_any(loc_match, axis=0)).numpy()

        tp_gt_box = gt_box[tf.reduce_any(loc_match, axis=0).numpy()]
        tp_gt_label = gt_label[tf.reduce_any(loc_match, axis=0).numpy()]
        fn_gt_box = gt_box[fn_bool]
        fn_gt_label = gt_label[fn_bool]

        tp_boxes.append(tp_gt_box)
        tp_labels.append(tp_gt_label)
        fn_boxes.append(fn_gt_box)
        fn_labels.append(fn_gt_label)

    try:
        tp_boxes = tf.concat(tp_boxes, axis=0)
        tp_labels = tf.concat(tp_labels, axis=0)
    except: 
        tp_boxes = tf.constant([[]], dtype=tf.float64)
        tp_labels = tf.constant([], dtype=tf.int64)

    try:
        fn_boxes = tf.concat(fn_boxes, axis=0)
        fn_labels = tf.concat(fn_labels, axis=0)
    except:
        fn_boxes = tf.constant([[]], dtype=tf.float64)
        fn_labels = tf.constant([], dtype=tf.int64)

    return tp_boxes, tp_labels, fn_boxes, fn_labels


def compute_iou(final_bbox, gt_box):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(final_bbox, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_box, 4, axis=-1)

    bbox_area = (bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1)
    gt_area = (gt_y2 - gt_y1) * (gt_x2 - gt_x1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [1, 0]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [1, 0]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [1, 0]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [1, 0]))

    inter_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    union_area = (bbox_area + tf.transpose(gt_area, [1, 0]) - inter_area)
    iou = inter_area / union_area

    return iou
#%%
# path argument
path = "/Users/wonhyung64/data/diagnosis"
total_labels = 21

#%%
# def data loader
samples = iter(os.listdir(path))
iou_thresh = 0.5
score_thresh = 0.3

total_fn = []

while True:
    sample = next(samples)
    try:
        with open(f"{path}/{sample}", encoding="UTF-8") as f:
            json_load = json.load(f)
    except: continue
    json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}

    gt_boxes = json_load["gt_boxes"]
    gt_labels = json_load["gt_labels"]
    roi_bboxes = json_load["roi_bboxes"]
    roi_scores = json_load["roi_scores"]
    pred_bboxes = json_load["pred_bboxes"]
    pred_labels = json_load["pred_labels"]
    final_bboxes = json_load["final_bboxes"]
    final_labels = json_load["final_labels"]
    final_scores = json_load["final_scores"]
    
    tp_boxes, tp_labels, fn_boxes, fn_labels = find_fn(gt_boxes, gt_labels, final_bboxes, final_labels, total_labels, iou_thresh=0.5)
    if tf.shape(tp_labels) != 0:
        tp_labels = tf.expand_dims(tp_labels, -1)
        tp = tf.concat([
            tf.zeros_like(tp_labels, dtype=tf.float64),
            tp_boxes,
            tf.cast(tp_labels, dtype=tf.float64)
            ], axis=-1)
        total_fn += tp.numpy().tolist()

    # fn mechanism
    if tf.shape(fn_labels) != 0:
        iou_pred_fn = compute_iou(pred_bboxes, fn_boxes)
        pred_iou_cond = iou_pred_fn >= iou_thresh

        # check cls 
        fn_cls_problem = tf.reduce_any(pred_iou_cond, axis=[0, 1])
        pred_cls_problem = tf.reduce_any(pred_iou_cond, axis=-1)

        fn_cls_problem_boxes = fn_boxes[fn_cls_problem]
        fn_cls_problem_labels = fn_labels[fn_cls_problem]

        fn_reg_problem_boxes = fn_boxes[tf.logical_not(fn_cls_problem)]
        fn_reg_problem_labels = fn_labels[tf.logical_not(fn_cls_problem)]


        # check cali, inter, bg
        s_loc = pred_labels[tf.logical_not(tf.reduce_any(pred_cls_problem, axis=-1))] # logical_not 지워야 됨

        for c, b in zip(fn_cls_problem_labels, fn_cls_problem_boxes):  #fn_labels_1 으로 바꾸기
            if tf.reduce_any(s_loc[..., c] >= score_thresh): 
                fn_cali = [1.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_cali]
            elif tf.reduce_any(s_loc[..., 0] >= score_thresh): 
                fn_bg = [3.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_bg]
            else: 
                fn_inter = [2.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_inter]

        iou_roi_fn = compute_iou(roi_bboxes, fn_reg_problem_boxes)
        # check regressor, proposal
        roi_iou_cond = iou_roi_fn >= iou_thresh
        fn_reg_bool = tf.reduce_any(roi_iou_cond, axis=0)

        fn_reg_box = fn_reg_problem_boxes[fn_reg_bool]
        fn_reg_label = tf.expand_dims(fn_reg_problem_labels[fn_reg_bool], axis=-1)
        if tf.shape(fn_reg_label)[0] != 0:
            fn_reg = tf.concat([
                tf.ones_like(fn_reg_label, dtype=tf.float64) * 4.,
                fn_reg_box,
                tf.cast(fn_reg_label, dtype=tf.float64)
                ], axis=-1)
            total_fn += fn_reg.numpy().tolist()

        fn_proposal_box = fn_reg_problem_boxes[tf.logical_not(fn_reg_bool)]
        fn_proposal_label = tf.expand_dims(fn_reg_problem_labels[tf.logical_not(fn_reg_bool)], axis=-1)
        if tf.shape(fn_proposal_label)[0] != 0:
            fn_proposal = tf.concat([
                tf.ones_like(fn_proposal_label, dtype=tf.float64) * 5.,
                fn_proposal_box,
                tf.cast(fn_proposal_label, dtype=tf.float64)
                ], axis=-1)
            total_fn += fn_proposal.numpy().tolist()
#%%
fn_df = pd.DataFrame(total_fn, columns=["type", "box_y1", "box_x1", "box_y2", "box_x2", "label"])

fn_df_all.to_csv(f"{path}/frcnn_fn_processed.csv", index=False)
pd.read_csv(f"{path}/frcnn_fn_processed.csv")
pd.read_

type_dict = {
    1.: "Calibration",
    2.: "Interclass",
    3.: "Background",
    4.: "Regressor",
    5.: "Proposal"
}

fn_df.loc[:, "type"] = fn_df["type"].map(lambda x: type_dict[x])


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(fn_df["type"], )

#%%
total_gt = []

while True:
    sample = next(samples)
    try:
        with open(f"{path}/{sample}", encoding="UTF-8") as f:
            json_load = json.load(f)
    except: continue
    json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}

    gt_boxes = json_load["gt_boxes"]
    gt_labels = json_load["gt_labels"]
    gt = np.concatenate([gt_boxes, np.expand_dims(gt_labels, -1)], axis=-1)
    total_gt += gt.tolist()

#%%
gt_df = pd.DataFrame(total_gt, columns=["box_y1", "box_x1", "box_y2", "box_x2", "label"])
gt_df.to_csv(f"{path}/pascal_gt_processed.csv", index=False)


#%%
sns.histplot(fn_df["type"])

tp_list = []
for c in range(1, 21):
    tp = len(gt_df[gt_df["label"] == c]) - len(fn_df[fn_df["label"] == c])
    print(f'Class {c} TP:  {tp}')
    tp_list.append(tp)


fn_df_all = pd.read_csv(f"{path}/frcnn_fn.csv")
gt_df = pd.read_csv(f"{path}/pascal_gt.csv")

fn_df = fn_df_all[fn_df_all["type"] == 5]


plt.bar(range(1, 21), tp_list)

fn_df["area"] = fn_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) * (x["box_x2"] - x["box_x1"]), axis=1)
gt_df["area"] = gt_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) * (x["box_x2"] - x["box_x1"]), axis=1)

fn_area = sns.histplot(fn_df["area"], kde="stat", color="b")
gt_area = sns.histplot(gt_df["area"], kde="stat", color="r")

fn_df["ratio"] = fn_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) / (x["box_x2"] - x["box_x1"]), axis=1)
gt_df["ratio"] = gt_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) / (x["box_x2"] - x["box_x1"]), axis=1)

fn_ratio = sns.histplot(fn_df["ratio"], kde="stat", color="b")
gt_ratio = sns.histplot(gt_df["ratio"], kde="stat", color="r")

gt_15 = gt_df[gt_df["label"] == 15.0]


plt.subplots(figsize=(10, 10), facecolor="white")
fn_class = sns.histplot(fn_df["label"].astype(int), kde="stat", color="b", alpha=0., edgecolor="white")
gt_class = sns.histplot(gt_df["label"].astype(int), kde="stat", color="r", alpha=0., edgecolor="white")

#%%
fig_area, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), facecolor="white")
sns.histplot(gt_df[gt_df["label"] == 13.0]["area"], stat="density", kde=True, color="r", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 14.0]["area"], stat="density", kde=True, color="orange", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 15.0]["area"], stat="density", kde=True, color="y", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 16.0]["area"], stat="density", kde=True, color="g", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 17.0]["area"], stat="density", kde=True, color="b", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 18.0]["area"], stat="density", kde=True, color="navy", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 19.0]["area"], stat="density", kde=True, color="purple", alpha=0., edgecolor="white", ax=axes[0])

sns.histplot(fn_df[fn_df["label"] == 13.0]["area"], stat="density", kde=True, color="r", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 14.0]["area"], stat="density", kde=True, color="orange", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 15.0]["area"], stat="density", kde=True, color="y", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 16.0]["area"], stat="density", kde=True, color="g", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 17.0]["area"], stat="density", kde=True, color="b", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 18.0]["area"], stat="density", kde=True, color="navy", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 19.0]["area"], stat="density", kde=True, color="purple", alpha=0., edgecolor="white", ax=axes[1])

# axes[0].set_yticks([0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.0])
# axes[1].set_yticks([0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.0])
axes[0].set_yticks(np.arange(0, 10.5, 0.5))
axes[1].set_yticks(np.arange(0, 10.5, 0.5))
axes[0].set_ylim([0, 10.0])
axes[1].set_ylim([0, 10.0])
#%%
fig_ratio, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), facecolor="white")
sns.histplot(gt_df[gt_df["label"] == 13.0]["ratio"], stat="density", kde=True, color="r", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 14.0]["ratio"], stat="density", kde=True, color="orange", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 15.0]["ratio"], stat="density", kde=True, color="y", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 16.0]["ratio"], stat="density", kde=True, color="g", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 17.0]["ratio"], stat="density", kde=True, color="b", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 18.0]["ratio"], stat="density", kde=True, color="navy", alpha=0., edgecolor="white", ax=axes[0])
sns.histplot(gt_df[gt_df["label"] == 19.0]["ratio"], stat="density", kde=True, color="purple", alpha=0., edgecolor="white", ax=axes[0])

sns.histplot(fn_df[fn_df["label"] == 13.0]["ratio"], stat="density", kde=True, color="r", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 14.0]["ratio"], stat="density", kde=True, color="orange", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 15.0]["ratio"], stat="density", kde=True, color="y", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 16.0]["ratio"], stat="density", kde=True, color="g", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 17.0]["ratio"], stat="density", kde=True, color="b", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 18.0]["ratio"], stat="density", kde=True, color="navy", alpha=0., edgecolor="white", ax=axes[1])
sns.histplot(fn_df[fn_df["label"] == 19.0]["ratio"], stat="density", kde=True, color="purple", alpha=0., edgecolor="white", ax=axes[1])

axes[0].set_yticks(np.arange(0, 1.41, 0.05))
axes[1].set_yticks(np.arange(0, 1.41, 0.05))
axes[0].set_ylim([0, 1.4])
axes[1].set_ylim([0, 1.4])
# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection":"3d"})

fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax.set_ylabel(ydata, fontdict=fontlabel, labelpad=16)
ax.set_title("Z", fontdict=fontlabel)
    
    ax.scatter(data["X"], data[ydata], data["Z"],  # 3D scatter plot
               c=data["Z"], cmap="inferno", s=5, alpha=0.5)
gt_df["ctr_x"] = gt_df["box_y2"] - gt_df["box_y1"]
gt_df["ctr_y"] = gt_df["box_x2"] - gt_df["box_x1"]

fn_df["ctr_x"] = fn_df["box_y2"] - fn_df["box_y1"]
fn_df["ctr_y"] = fn_df["box_x2"] - fn_df["box_x1"]

xx, yy = np.meshgrid(gt_df["ctr_x"], gt_df["ctr_y"])

#%%
fig_loc, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), facecolor="white") 
sns.scatterplot(gt_df[gt_df["label"] == 13]["ctr_x"], gt_df[gt_df["label"] == 13]["ctr_y"], color = "r", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 14]["ctr_x"], gt_df[gt_df["label"] == 14]["ctr_y"], color = "orange", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 15]["ctr_x"], gt_df[gt_df["label"] == 15]["ctr_y"], color = "y", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 16]["ctr_x"], gt_df[gt_df["label"] == 16]["ctr_y"], color = "g", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 17]["ctr_x"], gt_df[gt_df["label"] == 17]["ctr_y"], color = "b", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 18]["ctr_x"], gt_df[gt_df["label"] == 18]["ctr_y"], color = "navy", alpha=.5, ax=axes[0])
sns.scatterplot(gt_df[gt_df["label"] == 19]["ctr_x"], gt_df[gt_df["label"] == 19]["ctr_y"], color = "purple", alpha=.5, ax=axes[0])

sns.scatterplot(fn_df[fn_df["label"] == 13]["ctr_x"], fn_df[fn_df["label"] == 13]["ctr_y"], color = "r", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 14]["ctr_x"], fn_df[fn_df["label"] == 14]["ctr_y"], color = "orange", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 15]["ctr_x"], fn_df[fn_df["label"] == 15]["ctr_y"], color = "y", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 16]["ctr_x"], fn_df[fn_df["label"] == 16]["ctr_y"], color = "g", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 17]["ctr_x"], fn_df[fn_df["label"] == 17]["ctr_y"], color = "b", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 18]["ctr_x"], fn_df[fn_df["label"] == 18]["ctr_y"], color = "navy", alpha=.5, ax=axes[1])
sns.scatterplot(fn_df[fn_df["label"] == 19]["ctr_x"], fn_df[fn_df["label"] == 19]["ctr_y"], color = "purple", alpha=.5, ax=axes[1])

#%%
def build_anchors(feature_map_shape, anchor_scales, img_size, anchor_ratios) -> tf.Tensor:
    """
    generate reference anchors on grid

    Args:
        hyper_params (Dict): hyper parameters

    Returns:
        tf.Tensor: anchors
    """
    grid_map = build_grid(feature_map_shape[0])

    base_anchors = []
    for scale in anchor_scales:
        scale /= img_size[0]
        for ratio in anchor_ratios:
            w = tf.sqrt(scale**2 / ratio)
            h = w * ratio
            base_anchors.append([-h / 2, -w / 2, h / 2, w / 2])

    base_anchors = tf.cast(base_anchors, dtype=tf.float32)

    anchors = tf.reshape(base_anchors, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))

    anchors = tf.reshape(anchors, (-1, 4))
    anchors = tf.clip_by_value(t=anchors, clip_value_min=0, clip_value_max=1)

    return anchors


def build_grid(feature_map_shape):

    stride = 1 / feature_map_shape

    grid_coords_ctr = tf.cast(
        tf.range(0, feature_map_shape) / feature_map_shape + stride / 2,
        dtype=tf.float32,
    )

    grid_x_ctr, grid_y_ctr = tf.meshgrid(grid_coords_ctr, grid_coords_ctr)

    flat_grid_x_ctr, flat_grid_y_ctr = tf.reshape(grid_x_ctr, (-1,)), tf.reshape(
        grid_y_ctr, (-1,)
    )

    grid_map = tf.stack(
        [flat_grid_y_ctr, flat_grid_x_ctr, flat_grid_y_ctr, flat_grid_x_ctr], axis=-1
    )

    return grid_map

# %%
anchors = build_anchors(feature_map_shape=[31., 31.], anchor_scales=[64., 128., 256.], img_size=[500, 500], anchor_ratios=[1.0, 2., 1./2.])
# %%
def build_rpn_target(anchors, gt_boxes, gt_labels, feature_map_shape, anchor_scales, anchor_ratios):
    batch_size = 1
    anchor_count = len(anchor_scales) * len(anchor_ratios)
    total_pos_bboxes = 128
    total_neg_bboxes = 128
    variances = [0.1, 0.1, 0.2, 0.2]
    pos_threshold = 0.65
    neg_threshold = 0.25

    iou_map = compute_iou(tf.cast(anchors, tf.float64), gt_boxes)

    max_indices_each_row = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=-2, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)

    pos_mask = tf.greater(merged_iou_map, pos_threshold)

    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]

    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], -1)
    max_pos_mask = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=tf.fill((tf.shape(valid_indices)[0],), True),
        shape=tf.shape(pos_mask),
    )
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(
        pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32)
    )

    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count

    neg_mask = tf.logical_and(
        tf.less(merged_iou_map, neg_threshold), tf.logical_not(pos_mask)
    )
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)

    pos_labels = tf.where(
        pos_mask,
        tf.ones_like(pos_mask, dtype=tf.float32),
        tf.constant(-1.0, dtype=tf.float32),
    )

    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    gt_boxes_map = tf.gather(
        params=gt_boxes, indices=max_indices_each_row, batch_dims=1
    )

    expanded_gt_boxes = tf.where(
        tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map)
    )

    bbox_deltas = bbox_to_delta(anchors, expanded_gt_boxes) / variances

    bbox_deltas = tf.reshape(
        bbox_deltas,
        [batch_size] + feature_map_shape + [anchor_count * 4],
    )
    bbox_labels = tf.reshape(
        bbox_labels,
        [batch_size] + feature_map_shape + [anchor_count],
    )

    return bbox_deltas, bbox_labels
import matplotlib.pyplot as plt
fn_df.to_csv(f"{path}/pascal_frcnn.csv", index=False)
pd.read_csv(f"{path}/pascal_frcnn.csv")
plt.hist(fn_df["pos_iou"].map(lambda x: np.mean(x)))
plt.hist(fn_df["pos_iou"].map(lambda x: np.std(x)))

# %%
from tqdm import tqdm
gt_boxes = tf.expand_dims(gt_boxes, 0)
gt_boxes = tf.cast(gt_boxes, tf.float32)
gt_labels = tf.expand_dims(gt_labels, 0)

# tqdm.pandas()
# gt_df.progress_apply(lambda x: compute_pos_sample(anchors, tf.constant([[[x["box_y1"], x["box_x1"], x["box_y2"], x["box_x2"]]]], dtype=tf.float32), tf.constant([x["label"]], dtype=tf.int32)), 1)
fn_df
anchor_boxes
iou_pos_list = []
num_pos_list = []
for i in tqdm(range(len(fn_df))):
    y1, x1, y2, x2, label = fn_df.loc[i, "box_y1": "label"]
    gt_box = tf.constant([[[y1, x1, y2, x2]]], dtype=tf.float32)
    gt_label = tf.constant([[int(label)]], dtype=tf.int32)
    iou_pos = compute_pos_sample(anchors, gt_box, gt_label)
    num_pos = len(iou_pos)
    iou_pos_list.append(iou_pos.numpy().tolist())
    num_pos_list.append(num_pos)

gt_df.loc[:, "pos_num"] = num_pos_list
gt_df.loc[:, "pos_iou"] = iou_pos_list

fn_df.loc[:, "pos_num"] = num_pos_list
fn_df.loc[:, "pos_iou"] = iou_pos_list

def compute_pos_sample(anchors, gt_boxes, gt_labels):
    iou_map = generate_iou(anchors, gt_boxes)

    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    merged_iou_map = tf.reduce_max(iou_map, axis=2)

    pos_mask = tf.greater(merged_iou_map, pos_threshold)

    valid_indices_cond = tf.not_equal(gt_labels, -1)
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    valid_max_indices = max_indices_each_column[valid_indices_cond]

    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    max_pos_mask = tf.scatter_nd(
        indices=scatter_bbox_indices,
        updates=tf.fill((tf.shape(valid_indices)[0],), True),
        shape=tf.shape(pos_mask),
    )
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    
    pos_sample = merged_iou_map[pos_mask]

    return pos_sample


def generate_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    calculate Intersection over Union

    Args:
        anchors (tf.Tensor): reference anchors
        gt_boxes (tf.Tensor): bbox to calculate IoU

    Returns:
        tf.Tensor: Intersection over Union
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)

    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1]))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))

    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(
        y_bottom - y_top, 0
    )

    union_area = (
        tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area
    )

    return intersection_area / union_area

        with open(f"/Users/wonhyung64/data/diagnosis/sample0.json", encoding="UTF-8") as f:
            json_load = json.load(f)
        json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}

#%%
# path argument
path = "/Users/wonhyung64/data/diagnosis/retina"
total_labels = 20
# fn_df.to_csv("/Users/wonhyung64/data/diagnosis/pascal_frcnn.csv", index=False)
pd.read_csv("/Users/wonhyung64/data/diagnosis/pascal_retina.csv")
fn_df["pos_iou_mean"] = fn_df["pos_iou"].map(lambda x: np.mean(eval(x)))

len(iou)
#%%
# def data loader
samples = iter(os.listdir(path))
iou_thresh = 0.5
score_thresh = 0.3

total_fn = []

while True:
    sample = next(samples)
    try:
        with open(f"{path}/{sample}", encoding="UTF-8") as f:
            json_load = json.load(f)
    except: continue
    json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}

    gt_boxes = swap_xy(json_load["gt_boxes"])
    gt_labels = json_load["gt_labels"]
    anchor_boxes = json_load["anchor_boxes"]
    pred_bboxes = json_load["pred_bboxes"]
    pred_labels = json_load["pred_labels"]
    final_bboxes = json_load["final_bboxes"]
    final_labels = json_load["final_labels"]
    final_scores = json_load["final_scores"]
    
    tp_boxes, tp_labels, fn_boxes, fn_labels = find_fn(gt_boxes, gt_labels, final_bboxes, final_labels, total_labels, iou_thresh=0.5)
    if tf.shape(tp_labels) != 0:
        tp_labels = tf.expand_dims(tp_labels, -1)
        tp = tf.concat([
            tf.zeros_like(tp_labels, dtype=tf.float64),
            tp_boxes,
            tf.cast(tp_labels, dtype=tf.float64)
            ], axis=-1)
        total_fn += tp.numpy().tolist()

    # fn mechanism
    if tf.shape(fn_labels) != 0:
        iou_pred_fn = compute_iou(pred_bboxes, fn_boxes)
        pred_iou_cond = iou_pred_fn >= iou_thresh

        # check cls 
        fn_cls_problem = tf.reduce_any(pred_iou_cond, axis=[0])
        pred_cls_problem = tf.reduce_any(pred_iou_cond, axis=-1)

        fn_cls_problem_boxes = fn_boxes[fn_cls_problem]
        fn_cls_problem_labels = fn_labels[fn_cls_problem]

        fn_reg_problem_boxes = fn_boxes[tf.logical_not(fn_cls_problem)]
        fn_reg_problem_labels = fn_labels[tf.logical_not(fn_cls_problem)]


        # check cali, inter, bg
        s_loc = pred_labels[tf.logical_not(tf.reduce_any(pred_cls_problem, axis=-1))] # logical_not 지워야 됨

        for c, b in zip(fn_cls_problem_labels, fn_cls_problem_boxes): #fn_labels_1 으로 바꾸기
            if tf.reduce_any(s_loc[..., c] >= score_thresh): 
                fn_cali = [1.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_cali]
            elif tf.reduce_all(s_loc <= score_thresh): 
                fn_bg = [3.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_bg]
            else: 
                fn_inter = [2.] + b.numpy().tolist() + [tf.cast(c, dtype=tf.float64).numpy()]
                total_fn += [fn_inter]

        iou_roi_fn = compute_iou(anchor_boxes, fn_reg_problem_boxes*512)
        # check regressor, proposal
        roi_iou_cond = iou_roi_fn >= iou_thresh
        fn_reg_bool = tf.reduce_any(roi_iou_cond, axis=0)

        fn_reg_box = fn_reg_problem_boxes[fn_reg_bool]
        fn_reg_label = tf.expand_dims(fn_reg_problem_labels[fn_reg_bool], axis=-1)
        if tf.shape(fn_reg_label)[0] != 0:
            fn_reg = tf.concat([
                tf.ones_like(fn_reg_label, dtype=tf.float64) * 4.,
                fn_reg_box,
                tf.cast(fn_reg_label, dtype=tf.float64)
                ], axis=-1)
            total_fn += fn_reg.numpy().tolist()

        fn_proposal_box = fn_reg_problem_boxes[tf.logical_not(fn_reg_bool)]
        fn_proposal_label = tf.expand_dims(fn_reg_problem_labels[tf.logical_not(fn_reg_bool)], axis=-1)
        if tf.shape(fn_proposal_label)[0] != 0:
            fn_proposal = tf.concat([
                tf.ones_like(fn_proposal_label, dtype=tf.float64) * 5.,
                fn_proposal_box,
                tf.cast(fn_proposal_label, dtype=tf.float64)
                ], axis=-1)
            total_fn += fn_proposal.numpy().tolist()
#%%
fn_df = pd.DataFrame(total_fn, columns=["type", "box_y1", "box_x1", "box_y2", "box_x2", "label"])
fn_df["type"].value_counts()
fn_df.to_csv(f"{path}/")

#%%
anchor_boxes
gt_boxes
from tqdm import tqdm
iou_pos_list = []
num_pos_list = []
num_neg_list = []
for i in tqdm(range(len(fn_df))):

    y1, x1, y2, x2, label = fn_df.loc[i, "box_y1": "label"]
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

    y1, x1, y2, x2, label = fn_df.loc[i, "box_y1": "label"]
    gt_box = tf.constant([[y1, x1, y2, x2]], dtype=tf.float64)
    gt_label = tf.constant([[int(label)]], dtype=tf.int32)
    iou_pos = compute_pos_sample(anchors, gt_box, gt_label)
    num_pos = len(iou_pos)
    iou_pos_list.append(iou_pos.numpy().tolist())
    num_pos_list.append(num_pos)

gt_df.loc[:, "pos_num"] = num_pos_list
gt_df.loc[:, "pos_iou"] = iou_pos_list

fn_df.loc[:, "pos_num"] = num_pos_list
fn_df.loc[:, "pos_iou"] = iou_pos_list
