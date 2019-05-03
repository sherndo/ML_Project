import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid", font_scale=1.8, rc={"lines.linewidth": 4, "font.family":"Times New Roman"})

df = pd.read_csv('data/krieg_mm_results.csv')
df_new = pd.DataFrame(columns=['Test Year','Model','Accuracy','Log Loss'])

models = ['NB','LR','XGB','Seed']
metrics = ['Acc','Ll']
rows_to_exclude = ['Avg','STD']
years = [x for x in df['Test Year'].unique() if x not in rows_to_exclude]

for y, m in product(years, models):
    if m != 'Seed':
        logloss = df.loc[df['Test Year'] == y][m + ' ' + metrics[1]].values[0]
    else:
        logloss = 1
    df_new.loc[len(df_new.index)] = [y[:4], m, 
               df.loc[df['Test Year'] == y][m + ' ' + metrics[0]].values[0],
               logloss ]
        
    
sns.relplot(data=df_new, kind='line', x='Test Year',y='Accuracy', hue='Model', style='Model', height=10, aspect=2)
plt.savefig("figures/accall.png")
sns.catplot(data=df_new, x='Model', y='Accuracy', kind='violin', ci='sd', height=8, aspect=1.5)
plt.savefig("figures/accavg.png")
sns.relplot(data=df_new[df_new['Model'] != 'Seed'], kind='line', x='Test Year',y='Log Loss',hue='Model', style='Model', height=10, aspect=2)
plt.savefig("figures/loglossall.png")
sns.catplot(data=df_new[df_new['Model'] != 'Seed'], x='Model', y='Log Loss', kind='violin', ci='sd', height=8, aspect=1.5)
plt.savefig("figures/loglossavg.png")