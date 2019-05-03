# author: skrieg
# input: 3 csv files, 1) regular season stats by game, 2) regular season summary (SOS and adj win %), and 3) tourney seeding info
# output: a csv file listing all possible tourney matchups for a given year
#       the output is determined by looking at the tourney seedings and generating hypothetical matchups between all
#       64 teams in the tournament for that year.
#       Each row in the CSV represents one hypothetical matchup, and all the team stats that we will use for predicting the outcome.
#       Most of the rows will ultimately not be useful, since only 63 out of the 
#       (64 choose 2) possible games occur for a given year's tournament.

#   UPDATE 4/17/19 : Modified this script to include mirror matchups.
#   Each matchup now has TWO rows in the dataset. One with team a as T1 and team b as T2; and one for the mirror.
#   This is to balance out the statistical significance of our feature set.

import pandas as pd
from itertools import combinations

# calculate a given team's efficiency. 
# df columns must be in the following order: [Points Scored, FGA, OR, TO, FTA]
# efficiency is defined as (points scored / possession count)
def get_efficiency(df):
    return df[df.columns[0]].sum() / get_total_possessions(df)
    
# df columns must have the following labels: ['Assists', 'FGM']
def get_adj_efficiency(efficiency, df):
    return efficiency * (df[df.columns[0]].sum() / df[df.columns[1]].sum())

# (FGA – OR) + TO + (Y * FTA)
def get_total_possessions(df):
    return (df[df.columns[1]].sum() - df[df.columns[2]].sum()) + df[df.columns[3]].sum() + (0.44 * df[df.columns[4]].sum())

# standard deviation of point differential for each regular season game
# df columns must be in the following order: [Points Scored, Opp. Points]
def get_win_deviation(df):
    s = df[df.columns[0]] - df[df.columns[1]]
    return s.std()

    
# inf1 is team regular season stats by game
inf1 = 'data/season_results.csv'
# inf2 is team regular season summary (strength of schedule and adjusted win %)
inf2 = 'data/sos_results.csv'
# inf3 is tournament seeding info
inf3 = 'data/tourney_seeds.csv'

df_in1 = pd.read_csv(inf1)
df_in2 = pd.read_csv(inf2)
df_in3 = pd.read_csv(inf3)

eff_cols = ['Points','FGA','OR','TO','FTA']
adj_eff_cols = ['Ast','FGM']
opp_eff_cols = ['OpPoints','OpFGA','OpOR','OpTO','OpFTA']
opp_adj_eff_cols = ['OpAst','OpFGM']
win_dev_cols = ['Points','OpPoints']


final_cols = ['Year', 
             'T1 TeamID', 'T1 Seed','T1 SoS', 'T1 OE', 'T1 AdjOE', 'T1 DE', 'T1 AdjDE', 'T1 WinDev', 'T1 AdjWin',
             'T2 TeamID', 'T2 Seed','T2 SoS', 'T2 OE', 'T2 AdjOE', 'T2 DE', 'T2 AdjDE', 'T2 WinDev', 'T2 AdjWin',
             'Winner']
years_to_retrieve = range(2003,2019)
#years_to_retrieve = df_in3['Year'].unique()

# create new dataframe for output
df = pd.DataFrame(columns=final_cols)

# start with tournament seeds, since those area ll the teams whose matchups we need to predict
for y in sorted(years_to_retrieve):
    print('Starting year %s...' % y)
    df_y = df_in3.loc[df_in3[df_in3.columns[0]] == y]
    for t1, t2 in combinations(df_y[df_y.columns[2]].unique(), 2):
        matchup = {'Year': y, 'T1 TeamID': t1, 'T2 TeamID': t2,
                   'T1 Seed': df_y[df_y.columns[1]].loc[df_y[df_y.columns[2]] == t1].iloc[0],
                   'T2 Seed': df_y[df_y.columns[1]].loc[df_y[df_y.columns[2]] == t2].iloc[0]}
        
        mirror = {'Year': y, 'T1 TeamID': t2, 'T2 TeamID': t1,
                   'T1 Seed': df_y[df_y.columns[1]].loc[df_y[df_y.columns[2]] == t2].iloc[0],
                   'T2 Seed': df_y[df_y.columns[1]].loc[df_y[df_y.columns[2]] == t1].iloc[0]}
        
        # get team 1 stats
        df_in1_t1 = df_in1.loc[(df_in1[df_in1.columns[0]] == y) & (df_in1[df_in1.columns[1]] == t1)]
        matchup['T1 SoS'] = mirror['T2 SoS'] = df_in2[df_in2.columns[2]].loc[(df_in2[df_in2.columns[0]] == y) & (df_in2[df_in2.columns[1]] == t1)].iloc[0]
        matchup['T1 AdjWin'] = mirror['T2 AdjWin'] = df_in2[df_in2.columns[3]].loc[(df_in2[df_in2.columns[0]] == y) & (df_in2[df_in2.columns[1]] == t1)].iloc[0]
        matchup['T1 OE'] = mirror['T2 OE'] = get_efficiency(df_in1_t1[eff_cols])
        matchup['T1 AdjOE'] = mirror['T2 AdjOE'] = get_adj_efficiency(matchup['T1 OE'], df_in1_t1[adj_eff_cols])
        matchup['T1 DE'] = mirror['T2 DE'] = get_efficiency(df_in1_t1[opp_eff_cols])
        matchup['T1 AdjDE'] = mirror['T2 AdjDE'] = get_adj_efficiency(matchup['T1 DE'], df_in1_t1[opp_adj_eff_cols])
        matchup['T1 WinDev'] = mirror['T2 WinDev'] = get_win_deviation(df_in1_t1[win_dev_cols])
        
        # get team 2 stats
        df_in1_t2 = df_in1.loc[(df_in1[df_in1.columns[0]] == y) & (df_in1[df_in1.columns[1]] == t2)]
        matchup['T2 SoS'] = mirror['T1 SoS'] = df_in2[df_in2.columns[2]].loc[(df_in2[df_in2.columns[0]] == y) & (df_in2[df_in2.columns[1]] == t2)].iloc[0]
        matchup['T2 AdjWin'] = mirror['T1 AdjWin'] = df_in2[df_in2.columns[3]].loc[(df_in2[df_in2.columns[0]] == y) & (df_in2[df_in2.columns[1]] == t2)].iloc[0]
        matchup['T2 OE'] = mirror['T1 OE'] = get_efficiency(df_in1_t2[eff_cols])
        matchup['T2 AdjOE'] = mirror['T1 AdjOE'] = get_adj_efficiency(matchup['T1 OE'], df_in1_t2[adj_eff_cols])
        matchup['T2 DE'] = mirror['T1 DE'] = get_efficiency(df_in1_t2[opp_eff_cols])
        matchup['T2 AdjDE'] = mirror['T1 AdjDE'] = get_adj_efficiency(matchup['T1 DE'], df_in1_t2[opp_adj_eff_cols])
        matchup['T2 WinDev'] = mirror['T1 WinDev'] = get_win_deviation(df_in1_t2[win_dev_cols])
        
        df = df.append(matchup, ignore_index=True)
        df = df.append(mirror, ignore_index=True)

df.to_csv('data/tourney_possible_matchups_wmirrors.csv', index=False)