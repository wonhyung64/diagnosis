#%%
import os
import json
import numpy as np
import pandas as pd


#%%
df = pd.read_csv(f"./retinanet_r50_fpn_rsb-pretrain_1x_coco.csv")

df["area"] = df.apply(lambda x: (x["box_y2"] - x["box_y1"]) * (x["box_x2"] - x["box_x1"]), axis=1)
df["ratio"] = df.apply(lambda x: (x["box_y2"] - x["box_y1"]) / (x["box_x2"] - x["box_x1"]), axis=1)
df["ctr_x"] = df["box_y2"] - df["box_y1"]
df["ctr_y"] = df["box_x2"] - df["box_x1"]


