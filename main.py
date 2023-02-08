#%%
import os
import json
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from jax import grad, jit, vmap
from jax import random
from jax import device_put

# %%
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
%timeit jnp.dot(x, x.T).block_until_ready()

x = np.random.normal(size=(size, size)).astype(np.float32)
%timeit jnp.dot(x, x.T).block_until_ready()


x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
%timeit jnp.dot(x, x.T).block_until_ready()

#%%
# path argument
path = "/Users/wonhyung64/data/diagnosis"

# def data loader
samples = iter(os.listdir(path))

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

def calculate_pr(final_bbox, gt_box, mAP_threshold):
    bbox_num = tf.shape(final_bbox)[0]
    gt_num = tf.shape(gt_box)[0]

    true_pos = tf.Variable(tf.zeros(bbox_num))
    for i in range(bbox_num):
        bbox = tf.split(final_bbox, bbox_num, axis=1)[i]

        iou = generate_iou(bbox, gt_box)

        best_iou = tf.reduce_max(iou, axis=1)
        pos_num = tf.cast(tf.greater(best_iou, mAP_threshold), dtype=tf.float32)
        if tf.reduce_sum(pos_num) >= 1:
            gt_box = gt_box * tf.expand_dims(
                tf.cast(1 - pos_num, dtype=tf.float32), axis=-1
            )
            true_pos = tf.tensor_scatter_nd_update(true_pos, [[i]], [1])
    false_pos = 1.0 - true_pos
    true_pos = tf.math.cumsum(true_pos)
    false_pos = tf.math.cumsum(false_pos)

    recall = true_pos / gt_num
    precision = tf.math.divide(true_pos, true_pos + false_pos)

    return precision, recall


def calculate_ap_per_class(recall, precision):
    interp = tf.constant([i / 10 for i in range(0, 11)])
    AP = tf.reduce_max(
        [tf.where(interp <= recall[i], precision[i], 0.0) for i in range(len(recall))],
        axis=0,
    )
    AP = tf.reduce_sum(AP) / 11
    return AP

IOU_THRESH = 0.5
SCORE_THRESH = 0.3

def calculate_ap_const(
    final_bboxes, final_labels, gt_boxes, gt_labels, total_labels, mAP_threshold=0.5
):
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
            
        precision, recall = calculate_pr(final_bbox, gt_box, mAP_threshold)
        ap = calculate_ap_per_class(recall, precision)
        AP.append(ap)
    if AP == []:
        AP = 1.0
    else:
        AP = tf.reduce_mean(AP)
    return AP

