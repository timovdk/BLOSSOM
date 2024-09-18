import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from functools import partial
import gc

filenames = ['random_1', 'random_2', 'random_3', 'random_4', 'random_5', 'random_6', 'random_7', 'random_8', 'random_9', 'random_10', 'random_11', 'random_12', 'random_13', 'random_14', 'random_15', 'random_16', 'random_17', 'random_18', 'random_19', 'random_20', 'clustered_1', 'clustered_2', 'clustered_3', 'clustered_4', 'clustered_5', 'clustered_6', 'clustered_7', 'clustered_8', 'clustered_9', 'clustered_10', 'clustered_11', 'clustered_12', 'clustered_13', 'clustered_14', 'clustered_15', 'clustered_16', 'clustered_17', 'clustered_18', 'clustered_19', 'clustered_20']
path = "../blossom/hpc/outputs2/"
sample_times = [252, 415, 310, 192, 241, 308, 338, 231, 241, 233, 241, 330, 246, 246, 161, 327, 158, 218, 291, 431, 206, 185, 307, 258, 287, 244, 335, 291, 151, 206, 167, 356, 208, 224, 271, 181, 179, 279, 305, 246]
organism_group_labels = ["Bacteria", "Fungi", "Root-feeding Nematodes", "Bacterivorous Nematodes", "Fungivorous Nematodes", "Omnivorous Nematodes", "Fungivorous Mites", "Omnivorous Mites", "Collembolans"]
x_max = 400
y_max = 400

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use("seaborn-v0_8-whitegrid")

def rand_jitter(arr):
    if(len(arr)):
        stdev = .001 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev
    return arr


for idx, filename in enumerate(filenames):
    gc.collect()
    df = pd.read_csv(path + filename + ".csv")
    n = len(df["type"].unique())
    colors = colormaps['tab10'].colors

    counts_per_type = df.value_counts(['type', 'tick'])
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()

    for i in df['type'].unique():
        plt.plot(
            range(len(df["tick"].unique())),
            counts_per_type[i].reindex(range(len(df['tick'].unique())), fill_value=0).sort_index().to_list(),
            label=organism_group_labels[i],
            color=colors[i]
        )
    ax.set_xlim(0, df["tick"].max())
    ax.set_ylim(0, 150000)
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Agent Count")
    ax.set_title(f'{filename}, Counts')

    plt.legend()
    plt.savefig("./agent_counts_viz/" + filename + ".pdf", format="pdf", bbox_inches="tight")
    plt.close()
  
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    df1 = df[df["tick"] == 0]
    for t in np.unique(df1["type"]):
        subset_type = df1[df1['type'] == t]
        ax.scatter(rand_jitter(subset_type['x']), rand_jitter(subset_type['y']), color=colors[t], label=t, s=1, marker='s')
    ax.set_title(f'{filename}, Start')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig("./heatmaps/" + filename + "_hm_start.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    
    df1 = df[df["tick"] == df["tick"].max()]
    for t in np.unique(df1["type"]):
        subset_type = df1[df1['type'] == t]
        ax.scatter(rand_jitter(subset_type['x']), rand_jitter(subset_type['y']), color=colors[t], label=t, s=1, marker='s')
    ax.set_title(f'{filename}, End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig("./heatmaps/" + filename + "_hm_end.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    
    df1 = df[df["tick"] == sample_times[idx]]
    for t in np.unique(df1["type"]):
        subset_type = df1[df1['type'] == t]
        ax.scatter(rand_jitter(subset_type['x']), rand_jitter(subset_type['y']), color=colors[t], label=t, s=1, marker='s')
    ax.set_title(f'{filename}, End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig("./heatmaps/" + filename + "_hm_sample.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
    del df
    del df1
    gc.collect()
    
    #def update(t, ax):
    #    df1 = df[df["tick"] == t]
    #    ax.cla()
    #    for t in np.unique(df1["type"]):
    #        subset_type = df1[df1['type'] == t]
    #        if(len(subset_type['x']) != 0):
    #            ax.scatter(rand_jitter(subset_type['x']), rand_jitter(subset_type['y']), color=colors[t], label=t, s=10, marker='s')
    #    ax.set_title(f'{filename}')
    #    ax.set_xlabel('x')
    #    ax.set_ylabel('y')
    #    ax.set_xlim(0, x_max)
    #    ax.set_ylim(0, y_max)
    #    ax.grid(False)
    #    
    #fig = plt.figure(figsize=(15, 15))
    #ax = fig.add_subplot()
    #ani = FuncAnimation(fig, partial(update, ax=ax), frames=df["tick"].max(), interval=100)
    #ani.save("vids/" + filename + ".mp4")
    #plt.close()