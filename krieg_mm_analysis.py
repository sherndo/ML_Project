# Parameter log for mean log loss:
# Naive Bayes:
#   Default parameters: 0.706633
#
# Logistic Regression:
#   C=100.0: 0.550231
#   solver='newton-cg': 0.555598
#   Default parameters: 0.555604
#   solver='lbfgs': 0.559913
#   solver='sag' & max_iter=1000: 0.559996
#   solver='newton-cg', multi_class='multinomial': 0.583119
#
#   tol: no significant difference
#   penalty: no significant difference

#
# Gradient Boosting:
#   criterion='mse', n_estimators=45: 0.560244 
#   n_estimators=50: 0.561023
#   default: 0.568436
#   loss='exponential' = 0.575329

#
# XGBoost:
#   booster='dart': 0.561123
#   booster='gbtree': 0.563486
#   booster='gblinear': 0.563554
#
#   booster='dart' w/ mod eta: no significant difference
#   increasing gamma : bad
#   varying max_depth : bad
#   nothing else seemed to make a difference

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

df = pd.read_csv('tourney_matchup_results_wmirrors.csv')
cols_to_exclude = []
results = pd.DataFrame(columns=['Test Year','NB Acc','LR Acc', 'XGB Acc','NB Ll','LR Ll','XGB Ll'])
x = df.iloc[:, 0:len(df.columns)-1]
y = df.iloc[:, -1]

nb = GaussianNB()
lr = LogisticRegression(penalty='l1', tol=1e-5, C=100, solver='liblinear', max_iter=500)
xgb = XGBClassifier(booster='dart')

for year in x['Year'].unique():
    print(f'Testing year {year}')
    x_train = x[x['Year'] != year].drop(cols_to_exclude, axis=1)
    y_train = y.iloc[x_train.index]
    x_test = x[x['Year'] == year].drop(cols_to_exclude, axis=1)
    y_test = y.iloc[x_test.index]
    
    nb.fit(x_train, y_train)
    lr.fit(x_train, y_train)
    xgb.fit(x_train, y_train)
    
    results.loc[len(results.index)] = [year, 
                nb.score(x_test, y_test), 
                lr.score(x_test, y_test),
                xgb.score(x_test, y_test),
                log_loss(y_test, nb.predict_proba(x_test)), 
                log_loss(y_test, lr.predict_proba(x_test)), 
                log_loss(y_test, xgb.predict_proba(x_test)) ]

results.loc[len(results.index)] = ['Avg'] + [results[col].mean() for col in results.columns if col != 'Test Year']
results.loc[len(results.index)] = ['STD'] + [results[col].std() for col in results.columns if col != 'Test Year']
print(results[['Test Year'] + [x for x in results.columns if x.endswith('Ll')]])
print(results[['Test Year'] + [x for x in results.columns if x.endswith('Acc')]])
results.to_csv('krieg_mm_results.csv', index=False)