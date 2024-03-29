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
ex_files = [filename for filename in os.listdir() if filename.__contains__(".csv") and filename.__contains__("rsb") and filename.__contains__("test")]
# ex_files = [filename for filename in os.listdir() if filename.__contains__(".csv") and not filename.__contains__("test_")]
for i in tqdm(range(len(ex_files))):
    filename = ex_files[i]
    df = pd.read_csv(f"{filename}")
    for j in range(1,4):
        
        df_type = df[(df["type"] == 0) | (df["type"] == j)]
        df_type = df_type[df_type["area"] > 0.1]
        X = df_type.loc[:, "label":]
        # X = X.drop(columns=type_dict[j])
        y = df_type.loc[:, "type"].map(lambda x: 1. if x == j else x)

        model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        fig_local, fig_global = visualize_shap(shap_values, col_num=len(X.columns), filename=filename)

        fig_local.savefig(f"./ex7/{filename.split('.')[0]}_type_{j}_local.png")
        fig_global.savefig(f"./ex7/{filename.split('.')[0]}_type_{j}_global.png")


    X = df.loc[:, "label":]
    y = df.loc[:, "type"].map(lambda x: 1. if x >= 1 else x)

    model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X)
    shap_values = explainer(X)

    fig_local, fig_global = visualize_shap(shap_values, col_num=len(X.columns), filename=filename)

    shap.dependence_plot(("area", "label"), shap_interaction, X, display_features=X)
    shap.dependence_plot(("label", "area"), shap_interaction, X, display_features=X)
    shap.summary_plot(shap_interaction, X)

    fig_local.savefig(f"./ex6/{filename.split('.')[0]}_local.png")
    fig_global.savefig(f"./ex6/{filename.split('.')[0]}_global.png")

os.makedirs("ex6")


#%%
ex_files = [filename for filename in os.listdir() if filename.__contains__(".csv") and not filename.__contains__("test_")]
for i in tqdm(range(len(ex_files))):
    filename = ex_files[i]
    df_raw = pd.read_csv(f"{filename}")
    df_raw["type"] = df_raw.loc[:, "type"].map(lambda x: 1. if x >= 1 else x)
    df = df_raw.copy()
    X = df.loc[:, "label":]
    y = df.loc[:, "type"].map(lambda x: 1. if x >= 1 else x)

    model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_interaction = explainer.shap_interaction_values(X)

    '''absolute mean plot'''
    mean_shap = np.abs(shap_interaction).mean(0)
    df = pd.DataFrame(mean_shap, index=X.columns, columns = X.columns)

    df = df.where(df.values == np.diagonal(df), df.values*2)

    fig = plt.figure(figsize=(35, 20), facecolor="#002637", edgecolor="r")
    ax = fig.add_subplot()
    sns.heatmap(df.round(decimals=3), cmap="coolwarm", annot=True, fmt=".6g", cbar=False, ax=ax, annot_kws={"size": 30})
    ax.tick_params(axis='x', colors='w', labelsize=30)
    ax.tick_params(axis='y', colors='w', labelsize=30)

    plt.suptitle("SHAP interaction values", color="white", fontsize=60, y=0.97)
    plt.yticks(rotation=0) 
    plt.show()
    fig.savefig(f"./ex3/{filename.split('.')[0]}_inter.png")
fig
    '''Feature interaction analysis'''
#plot feature interaction
from matplotlib import gridspec
f1="area"
f2="label"
#plot function
plt.style.use("cyberpunk")

def plot_feature_interaction(f1, f2):
    # dependence plot
    fig = plt.figure(tight_layout=True, figsize=(20,10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)


    ax0 = fig.add_subplot(spec[0, 0])
    shap.dependence_plot(f1, shap_values, X, display_features=X, interaction_index=None, ax=ax0, show=False)
    ax0.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax0.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax0.set_title(f'SHAP main effect', fontsize=10)

    ax1 = fig.add_subplot(spec[0, 1])
    shap.dependence_plot((f1, f2), shap_interaction, X, display_features=X, ax=ax1, axis_color='w', show=False)
    ax1.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax1.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax1.set_title(f'SHAP interaction effect', fontsize=10)

    ax2 = fig.add_subplot(spec[1, 0])
    sns.scatterplot(x=f1, y=f2, data=df_raw, hue="type", ax=ax2, s=2)
    ax2.text(-1.5, -5, "1", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")
    ax2.text(0, 1, "2", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")
    ax2.text(1, 7, "3", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")

    ax2.set_title(f'scatter plot', fontsize=10)

    temp = pd.DataFrame({f1: df_raw[f1].values,'target': df_raw.type.values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)
    
    ax3 = fig.add_subplot(spec[1, 1])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(15000, center=True).mean(), data=temp, ax=ax3, s=2)
    ax3.set_title('How the target probability depends on f_02', fontsize=10)
    
    plt.suptitle("Feature Interaction Analysis\n f_02 and f_21", fontsize=30, y=1.15)
    plt.show()


f1='f_24'
f2='f_30'
plot_feature_interaction(f1, f2)
#%%
for i in tqdm(range(len(ex_files))):
    filename = ex_files[i]
    df_raw = pd.read_csv(f"{filename}")
    print(ex_files[i])
    print(df_raw["type"].value_counts())

#%%
import torch
import seaborn as sns

file = "retinanet_r18_fpn_1x_coco.csv"
df = pd.read_csv(f"{file}")

fig, axes = plt.subplots(ncols=2, nrows=8, figsize=(10, 40))
df["type"] = df["type"].map(lambda x: 1. if x >= 1 else 0.)
i = 0
sns.histplot(data=df, x="label", bins=80, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="fg_bg_ratio", bins=30, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="iou_mean", bins=30, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="iou_std", bins=30, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="area", bins=40, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="ratio", bins=40, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="ctr_x", bins=40, hue="type", multiple="stack", ax=axes[i][0])
i += 1
sns.histplot(data=df, x="ctr_y", bins=40, hue="type", multiple="stack", ax=axes[i][0])

test_df = pd.read_csv(f"test_{file}")
test_df["ctr_x"] = (test_df["box_x1"] + test_df["box_x2"]) / 2
test_df["ctr_y"] = (test_df["box_y1"] + test_df["box_y2"]) / 2

test_df["type"] = test_df["type"].map(lambda x: 1. if x >= 1 else 0.)
i = 0
sns.histplot(data=test_df, x="label", bins=80, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="fg_bg_ratio", bins=30, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="iou_mean", bins=30, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="iou_std", bins=30, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="area", bins=40, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="ratio", bins=40, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="ctr_x", bins=40, hue="type", multiple="stack", ax=axes[i][1])
i += 1
sns.histplot(data=test_df, x="ctr_y", bins=40, hue="type", multiple="stack", ax=axes[i][1])
fig

sns.boxplot(data=df, x="label")

small_df = df[df["area"] <= 0.1].reset_index(drop=True)
sns.histplot(data=small_df, x="area", bins=80, hue="type", multiple="stack")
sns.histplot(data=small_df, x="fg_bg_ratio", bins=80, hue="type", multiple="stack")
df

test_df = pd.read_csv(f"test_{file}")
test_df["type"] = test_df["type"].map(lambda x: 1. if x >= 1 else x)
small_test_df = test_df[test_df["area"] <= 0.1].reset_index(drop=True)
sns.histplot(data=small_test_df, x="area", bins=80, hue="type", multiple="stack")
sns.histplot(data=small_df, x="fg_bg_ratio", bins=80, hue="type", multiple="stack")
sns.histplot(data=small_df, x="fg_bg_ratio", bins=80, hue="type", multiple="stack")


#%%
flip_df = pd.read_csv("e6_test_retinanet_r50_fpn_rsb-pretrain_1x_coco.csv")
no_flip_df = pd.read_csv("e12_test_no_flip_retinanet_r50_fpn_rsb-pretrain_1x_coco.csv")
no_flip_df["label"].value_counts()
cls_df = no_flip_df[no_flip_df["type"] < 2.]

sns.histplot(data=cls_df, x="fg_bg_ratio", hue="type", multiple="fill")
sns.histplot(data=cls_df, x="label", hue="type", multiple="fill")
sns.histplot(data=cls_df, x="iou_mean", hue="type", multiple="fill")
sns.histplot(data=cls_df, x="iou_std", hue="type", multiple="fill")
sns.histplot(data=cls_df, x="area", hue="type", bins=100, multiple="fill")
sns.histplot(data=cls_df, x="ratio", hue="type", bins=50, multiple="fill")
sns.histplot(data=cls_df, x="ctr_x", hue="type", bins=50, multiple="fill")
sns.histplot(data=cls_df, x="ctr_y", hue="type", bins=50, multiple="fill")
cls_df["fg_bg_ratio"].value_counts()

#%%
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

test_df["ctr_x"] = (test_df["box_x2"] + test_df["box_x1"]) / 2
test_df["ctr_y"] = (test_df["box_y2"] + test_df["box_y1"]) / 2

#%% two-layers
node_names = ["type", "fg_bg_ratio", "iou_mean", "iou_std", "label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)

nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()

'''default settings~'''
# forbid type -> node edges
for i in range(1, len(nodes)):
    bk.add_forbidden_by_node(nodes[0], nodes[i])
    
# forbid model_node -> property_node edges
for i in range(1, 4):
    for j in range(4, len(nodes)): 
        bk.add_forbidden_by_node(nodes[i], nodes[j])
'''~default settings'''

cg2 = pc(data, 0.0001, background_knowledge=bk, node_names=node_names)
cg2.draw_pydot_graph()



#%% only-property
node_names = ["type", "label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)

nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()
'''default settings~'''
# forbid type -> node edges
for i in range(1, len(nodes)):
    bk.add_forbidden_by_node(nodes[0], nodes[i])
'''~default settings'''

cg2 = pc(data, background_knowledge=bk, node_names=node_names)
cg2.draw_pydot_graph()

#%% two-layers without inter
node_names = ["type", "fg_bg_ratio", "iou_mean", "iou_std", "label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)

nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()
'''default settings~'''
# forbid type -> node edges
for i in range(1, len(nodes)):
    bk.add_forbidden_by_node(nodes[0], nodes[i])
    
# forbid model_node -> property_node edges
for i in range(1, 4):
    for j in range(4, len(nodes)): 
        bk.add_forbidden_by_node(nodes[i], nodes[j])
'''~default settings'''

# forbid edges between model_nodes
for i in range(1, 4):
    for j in range(1, 4):
        if i == j: continue
        bk.add_forbidden_by_node(nodes[i], nodes[j])

#forbid edges between property_nodes
# for i in range(4, 9):
#     for j in range(4, 9):
#         if i == j: continue
#         bk.add_forbidden_by_node(nodes[i], nodes[j])

cg2 = pc(data, 0.0001, background_knowledge=bk, node_names=node_names)
cg2.draw_pydot_graph()

#%% two-layers without jump
node_names = ["type", "fg_bg_ratio", "iou_mean", "iou_std", "label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)

nodes = cg.G.get_nodes()
'''default settings~'''
# forbid type -> node edges
for i in range(1, len(nodes)):
    bk.add_forbidden_by_node(nodes[0], nodes[i])
    
# forbid model_node -> property_node edges
for i in range(1, 4):
    for j in range(4, len(nodes)): 
        bk.add_forbidden_by_node(nodes[i], nodes[j])
'''~default settings'''

# forbid property_node -> type edges
for i in range(4, 9):
    bk.add_forbidden_by_node(nodes[i], nodes[0])

cg2 = pc(data, background_knowledge=bk, node_names=node_names)
cg2.draw_pydot_graph()


#%% two-layers without inter and jump
node_names = ["type", "fg_bg_ratio", "iou_mean", "iou_std", "label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)

nodes = cg.G.get_nodes()
bk = BackgroundKnowledge()
'''default settings~'''
# forbid type -> node edges
for i in range(1, len(nodes)):
    bk.add_forbidden_by_node(nodes[0], nodes[i])
    
# forbid model_node -> property_node edges
for i in range(1, 4):
    for j in range(4, len(nodes)): 
        bk.add_forbidden_by_node(nodes[i], nodes[j])
'''~default settings'''

# forbid edges between model_nodes
for i in range(1, 4):
    for j in range(1, 4):
        if i == j: continue
        bk.add_forbidden_by_node(nodes[i], nodes[j])

#forbid edges between property_nodes
for i in range(4, 9):
    for j in range(4, 9):
        if i == j: continue
        bk.add_forbidden_by_node(nodes[i], nodes[j])

# forbid property_node -> type edges
for i in range(4, 9):
    bk.add_forbidden_by_node(nodes[i], nodes[0])

cg2 = pc(data, 0.0000001, background_knowledge=bk, node_names=node_names)
cg2.draw_pydot_graph()

# %%
node_names = ["label", "area", "ratio", "ctr_x", "ctr_y"]
data = test_df[node_names].to_numpy()
cg = pc(data, node_names=node_names)
cg.draw_pydot_graph()

#%%
tmp = test_df.groupby("label").mean()["ctr_y"].reset_index()
sns.barplot(tmp, y="ctr_y", x="label")
tmp.sort_values("ctr_y", ascending=False).head(10)
tmp.sort_values()

CLASSES
tmp["label"] = tmp["label"].map(lambda x: CLASSES[int(x)])
