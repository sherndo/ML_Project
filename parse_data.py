import pandas as pd
import os, sys, collections
from math import log2 as log

def parse_data(season_file, teams_file, sos_file):
    # read csv into panda dataframe using headers and found delimiter
    season_data = pd.read_csv(season_file, delimiter=',')
    team_info = pd.read_csv(teams_file, delimiter=',', encoding="latin1")
    sos_info = pd.read_csv(sos_file, delimiter=',')
    
    team_data_cols = [
        "Year",
        "TeamID",
        "GameNum",
        "Points",
        "FGA",
        "FGM",
        "FTA",
        "OR",
        "TO",
        "Ast",
        "OpPoints",
        "OpFGA",
        "OpFGM",
        "OpFTA",
        "OpOR",
        "OpTO",
        "OpAst",
        "Result"
    ]

    team_data = pd.DataFrame(columns=team_data_cols)
    year = season_data.iloc[0].Season
    game_count = collections.Counter()
    win_percentage = {}
    i = 0
    for index, game in season_data.iterrows():
        if year != game.Season:
            game_count = collections.Counter()
        
        if game.Season not in win_percentage:
            win_percentage[game.Season] = {}
        
        if game.WTeamID not in win_percentage[game.Season]:
            win_percentage[game.Season][game.WTeamID] = [0,0]
        if game.LTeamID not in win_percentage[game.Season]:
            win_percentage[game.Season][game.LTeamID] = [0,0]
        
        row1 = [game.Season, game.WTeamID, game_count[game.WTeamID]+1, game.WScore, game.WFGA, game.WFGM, game.WFTA, game.WOR, game.WTO, game.WAst, game.LScore, game.LFGA, game.LFGM, game.LFTA, game.LOR, game.LTO, game.LAst, 1]
        row2 = [game.Season, game.LTeamID, game_count[game.LTeamID]+1, game.LScore, game.LFGA, game.LFGM, game.LFTA, game.LOR, game.LTO, game.LAst, game.WScore, game.WFGA, game.WFGM, game.WFTA, game.WOR, game.WTO, game.WAst, 0]

        team_data.loc[i] = row1
        team_data.loc[i+1] = row2

        game_count[game.WTeamID] += 1
        game_count[game.LTeamID] += 1
        
        win_percentage[game.Season][game.WTeamID][0] += log(game_count[game.WTeamID])
        win_percentage[game.Season][game.WTeamID][1] += log(game_count[game.WTeamID])
        win_percentage[game.Season][game.LTeamID][1] += log(game_count[game.LTeamID])

        i += 2
        year = game.Season

    team_data.to_csv("/home/sherndo/Documents/Classes/MachineLearning/ML_Project/season_results_2019.csv", index=False)
    sos_cols = [
        "Year",
        "TeamID",
        "SOS",
        "AdjWin"
    ]
    sos_data = pd.DataFrame(columns=sos_cols)
    i = 0
    bad = set()
    for index, item in sos_info.iterrows():
        tid = team_info.loc[team_info['TeamNameSpelling'] == item.SCHOOL.lower()].TeamID
        if tid.empty:
            tid = team_info.loc[team_info['TeamNameSpelling'] == item.SCHOOL.lower().replace(".", "")].TeamID
        if tid.empty:
            tid = team_info.loc[team_info['TeamNameSpelling'] == item.SCHOOL.lower().replace("-", " ")].TeamID
        if tid.empty:
            tid = team_info.loc[team_info['TeamNameSpelling'] == item.SCHOOL.lower().replace(".", "").replace("-", " ")].TeamID
        
        tid = int(tid)
        wp = win_percentage[item.YEAR][tid][0] / win_percentage[item.YEAR][tid][1]
        row = [item.YEAR, tid, item.SOS, wp]
        sos_data.loc[i] = row
        i += 1

    sos_data.to_csv("/home/sherndo/Documents/Classes/MachineLearning/ML_Project/sos_results_2019.csv", index=False)

    return


if __name__=="__main__":
    season_file = "/home/sherndo/Documents/Classes/MachineLearning/ML_Project/mens-machine-learning-competition-2019/RegularSeasonDetailedResults.csv"
    team_file = "/home/sherndo/Documents/Classes/MachineLearning/ML_Project/mens-machine-learning-competition-2019/TeamSpellings.csv"
    sos_file = "/home/sherndo/Documents/Classes/MachineLearning/ML_Project/mens-machine-learning-competition-2019/StrengthOfSchedule.csv"

    parse_data(season_file, team_file, sos_file)