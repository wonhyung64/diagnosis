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

        fn_gt_box = gt_box[fn_bool]
        fn_gt_label = gt_label[fn_bool]

        fn_boxes.append(fn_gt_box)
        fn_labels.append(fn_gt_label)

    fn_boxes = tf.concat(fn_boxes, axis=0)
    fn_labels = tf.concat(fn_labels, axis=0)

    return fn_boxes, fn_labels


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

total_fn = []

while True:
    sample = next(samples)
    with open(f"{path}/{sample}", encoding="UTF-8") as f:
        json_load = json.load(f)
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

    fn_boxes, fn_labels = find_fn(gt_boxes, gt_labels, final_bboxes, final_labels, total_labels, iou_thresh=0.5)

    # fn mechanism
    iou_thresh = 0.5
    score_thresh = 0.3

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