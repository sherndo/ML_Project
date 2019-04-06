# just getting started...

import numpy as np
import pandas as pd
from itertools import combinations

# inf1 is team regular season stats by game
inf1 = 'dat_test/csv1.sample.csv'
# inf2 is team regular season summary (strength of schedule and adjusted win %)
inf2 = 'dat_test/csv2.sample.csv'
# inf3 is tournament seeding info
inf3 = 'dat_test/csv3.sample.csv'

df_in1 = pd.read_csv(inf1)
df_in2 = pd.read_csv(inf2)
df_in3 = pd.read_csv(inf3)

final_cols = ['Year', 
             'T1 TeamID', 'T1 Seed','T1 SoS', 'T1 OE', 'T1 DE', 'T1 Win Dev', 'T1 WWin %',
             'T2 TeamID', 'T2 Seed','T2 SoS', 'T2 OE', 'T2 DE', 'T2 Win Dev', 'T2 WWin %',
             'Winner']
#years_to_retrieve = range(2008,2009)
years_to_retrieve = df_in3['Year'].unique()

df = pd.DataFrame(columns=final_cols)

print(df_in1)
print(df_in2)
print(df_in3)
print(df)


for y in sorted(years_to_retrieve):
    df_y = df_in3.loc[df_in3['Year'] == y]
    for t1, t2 in combinations(df_y['TeamID'].unique(), 2):
        matchup = {'Year': y, 'T1 TeamID': t1, 'T2 TeamID': t2,
                   'T1 Seed': df_y['Seed'].loc[df_y['TeamID'] == t1].iloc[0],
                   'T2 Seed': df_y['Seed'].loc[df_y['TeamID'] == t2].iloc[0]}
        print(matchup)
        df = df.append(matchup, ignore_index=True)
        
print(df)