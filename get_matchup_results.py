# author: skrieg
# input1: tourney_possible_matchups.csv - a csv listing possible tourney matchups and team stats
# input2: ncaatourneycompactresults.csv - a csv listing the results of tourney matches that actually happened
# output: tourney_matchup_results.csv - a csv with the real match results from input2 joined to the team stats from input1

import pandas as pd
import string

df_matchups = pd.read_csv('tourney_possible_matchups.csv')
df_results = pd.read_csv('NCAATourneyCompactResults.csv')

def get_winner(df):
    y = df.iloc[0]
    t1 = df.iloc[1]
    t2 = df.iloc[2]
    
    result = df_results.loc[(df_results['Season'] == y) & 
                            (((df_results['WTeamID'] == t1) & (df_results['LTeamID'] == t2))
                            | ((df_results['WTeamID'] == t2) & (df_results['LTeamID'] == t1)))]
    if result.empty:
        return 'N/A'
    elif result['WTeamID'].iloc[0] == t1:
        return 'T1'
    else:
        return 'T2'
    

print(df_matchups)

# make seed an integer
df_matchups['T1 Seed'] = df_matchups['T1 Seed'].map(lambda x: x.strip(string.ascii_letters))
df_matchups['T2 Seed'] = df_matchups['T2 Seed'].map(lambda x: x.strip(string.ascii_letters))

# determine the winner
df_matchups['Winner'] = df_matchups[['Year','T1 TeamID','T2 TeamID']].apply(get_winner, axis=1)

# matchups with N/A result never happened, so don't include them
df_matchups.loc[df_matchups['Winner'] != 'N/A'].to_csv('tourney_matchup_results.csv', index=False)