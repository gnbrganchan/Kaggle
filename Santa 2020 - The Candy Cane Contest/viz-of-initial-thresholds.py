import pandas as pd
import numpy as np
from statistics import mean
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', None)
##

!pip install kaggle-environments --upgrade

from kaggle_environments import evaluate, make, utils

env = make("mab", debug=True)

N = 300
freq = [0] * 100
df = pd.DataFrame()
count_m = []
# repeat simulation N times
for i in tqdm(range(N)):
    env.reset()
    env.run(["submission1.py", "submission2.py"])
    df_tmp = pd.DataFrame({"trial" : [i+1]*100,"Machine" : list(range(100)), "threshold" : env.__dict__['steps'][0][0]['observation']['thresholds']})
    df = df.append(df_tmp)
    count_m.append(mean(env.__dict__['steps'][0][0]['observation']['thresholds']))
##

# initial threshold of each machine
sns.set()
plt.figure(figsize=(20,10))
sns.swarmplot(x="Machine", y="threshold", data=df)
##

# box plot of initial thresholds
plt.figure(figsize=(20,10))
sns.boxplot(x="Machine", y="threshold", data=df)
##

# correlation between mechines
cor_mat = np.corrcoef(df.pivot(values=['threshold'], index = ["trial"], columns=['Machine']).T)
plt.figure(figsize=(15,15))
sns.set(color_codes=True, font_scale=1.2)

ax = sns.heatmap(
    cor_mat, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right'
)

plt.show()
##

# histgram of mean of initial thresholds
plt.figure(figsize=(20,10))
sns.distplot(count_m)
