#%%
import os
import torch
import numpy as np
import pandas as pd

# %%
filenames = [filename for filename in os.listdir() if filename.__contains__("csv") and filename.__contains__("e12")]
df = pd.read_csv(filenames[0])
df["type"].value_counts()
df_loc = df[df["type"] != 3.]
df_loc["type"] = df_loc["type"].map(lambda x: 1. if x == 2. else 0.)






