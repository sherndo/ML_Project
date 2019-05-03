import pandas as pd
import os, sys
import pickle,numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import log_loss

def load_data(file_name):
    # read csv into panda dataframe using headers and found delimiter
    tourney_data = pd.read_csv(file_name, delimiter=',')

    all_vals = ['Year','T1 TeamID','T1 Seed','T1 SoS','T1 OE','T1 AdjOE','T1 DE','T1 AdjDE','T1 WinDev','T1 AdjWin','T2 TeamID','T2 Seed','T2 SoS','T2 OE','T2 AdjOE','T2 DE','T2 AdjDE','T2 WinDev','T2 AdjWin','Winner']
    select_vals = ['T1 Seed','T1 SoS','T1 OE','T1 AdjOE','T1 DE','T1 AdjDE','T1 AdjWin','T2 Seed','T2 SoS','T2 OE','T2 AdjOE','T2 DE','T2 AdjDE','T2 AdjWin','Winner']

    tourney_data = tourney_data.loc[:, select_vals]

    for row in tourney_data.iterrows():
        if row[1].loc['T1 Seed'] > row[1].loc['T2 Seed']:
            print("Yes")

    return tourney_data.iloc[:,:-1], tourney_data.iloc[:,-1]

def trainXGBoost(data, classes, k_fold, scoring):
    params = {
    "learning_rate"    : [0.01, 0.10, 0.30 ] ,
    "max_depth"        : [3, 7, 10, 15],
    "random_state": [0, 3, 7, 42, 145],
    "min_child_weight" : [ 1, 3, 7 ],
    "gamma"            : [ 0.0, 0.1, 0.2 , 0.4 ],
    "colsample_bytree" : [ 0.3, 0.4 , 0.5, 0.7 ],
    'subsample' : [ 0.7, 0.9, 1 ],
    "booster" : ['gbtree', 'gblinear', 'dart']
    }

    gbc = XGBClassifier(n_jobs=8,random_state=7, max_depth=3, booster='gbtree', learning_rate=0.1, colsample_bytree=0.3, gamma=0.2, min_child_weight=1, subsample=0.7)
    # {'booster': 'gbtree', 'colsample_bytree': 0.3, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'random_state': 7, 'subsample': 0.7}

    glf = GridSearchCV(gbc, params,cv=k_fold, scoring=scoring,refit='loss', return_train_score=True)
    glf.fit(data,classes)
    print(glf.best_params_)
    results = glf.cv_results_
    with open('archive/xgb_results.pkl', 'wb') as fp:
        pickle.dump(results,fp)

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV Evalution for Loss: XGBoost",
            fontsize=16)

    plt.xlabel('Evaluation Number')
    plt.ylabel("Loss")
    ax.set_ylim(0.45, 1)

    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = numpy.array(range(len(results['param_'+params.keys()[0]].data)), dtype=float)

    scorer='loss'
    color='g'
    sample = 'test'
    style = '-'
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = results['std_%s_%s' % (sample, scorer)]
    sample_score_mean *= -1
    sample_score_std *= -1
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = numpy.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.savefig('figures/xg_param_search_loss.png')

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV Evalution for Accuracy: XGBoost" ,
            fontsize=16)

    plt.xlabel('Evaluation Number')
    plt.ylabel("Accuracy")
    ax.set_ylim(0.45, 1)

    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = numpy.array(range(len(results['param_'+params.keys()[0]].data)), dtype=float)

    scorer='accuracy'
    color='g'
    sample = 'test'
    style = '-'
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = results['std_%s_%s' % (sample, scorer)]
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = numpy.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.savefig('figures/xg_param_search_accuracy.png')

    gbc_scores = cross_validate(gbc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)


    for i in range(len(gbc_scores['estimator'])):
        plot_importance(gbc_scores['estimator'][i], importance_type='cover')
        plt.savefig('figures/xg_'+str(i)+'_features.png')

    return gbc_scores

def trainLogisticRegression(data, classes, k_fold, scoring):
    lrc = LogisticRegression(tol=0.001, C=1000, max_iter=100,solver='lbfgs')
#     {'C': 1000, 'max_iter': 100, 'penalty': 'l2', 'random_state': 0, 'solver': 'liblinear', 'tol': 0.001}


    params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100,500,1000],
    'penalty': ['l1', 'l2'],
    "random_state": [0, 1, 3, 7, 9, 10, 42]
    }

    clf = GridSearchCV(lrc, params,cv=k_fold,scoring=scoring,refit='loss', return_train_score=True)
    clf.fit(data,classes)
    print(clf.best_params_)
    results = clf.cv_results_
    with open('archive/lr_results.pkl', 'wb') as fp:
        pickle.dump(results,fp)

    # for key in params.keys():
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV Evalution for Loss: Logistic Regression" ,
            fontsize=16)

    plt.xlabel('Evaluation Number')
    plt.ylabel("Loss")
    ax.set_ylim(0.45, 1)

    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = numpy.array(range(len(results['param_'+params.keys()[0]].data)), dtype=float)

    scorer='loss'
    color='g'
    sample = 'test'
    style = '-'
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = results['std_%s_%s' % (sample, scorer)]
    sample_score_mean *= -1
    sample_score_std *= -1
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = numpy.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.savefig('figures/lr_param_search_loss.png')

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV Evalution for Accuracy: Logistic Regression" ,
            fontsize=16)

    plt.xlabel('Evaluation Number')
    plt.ylabel("Accuracy")
    ax.set_ylim(0.45, 1)

    ax = plt.gca()

    # Get the regular numpy array from the MaskedArray
    X_axis = numpy.array(range(len(results['param_tol'].data)), dtype=float)

    scorer='accuracy'
    color='g'
    sample = 'test'
    style = '-'
    sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
    sample_score_std = results['std_%s_%s' % (sample, scorer)]
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == 'test' else 0, color=color)
    ax.plot(X_axis, sample_score_mean, style, color=color,
            alpha=1 if sample == 'test' else 0.7,
            label="%s (%s)" % (scorer, sample))

    best_index = numpy.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.savefig('figures/lr_param_search_accuracy.png')

    lrc_scores = cross_validate(lrc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(lrc_scores)

    return lrc_scores

def trainNaiveBayes(data, classes, k_fold, scoring):
    nbc = GaussianNB()

    nbc_scores = cross_validate(nbc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)

    return nbc_scores

if __name__=="__main__":
    file_name = "/".join(__file__.split("/")[:-1])
    print(file_name)
    if file_name == "":
        file_name = "data/tourney_matchup_results_wmirrors.csv"
    else:
        file_name += "/data/tourney_matchup_results_wmirrors.csv"

    scoring = {
        'accuracy': 'accuracy',
        'loss': 'neg_log_loss',
        'roc_auc': 'roc_auc'
    }

    plt.rcParams.update({'font.size': 16})

    data,classes = load_data(file_name)
    k_fold = StratifiedKFold(16)

    lr = trainLogisticRegression(data, classes, k_fold, scoring)
    xg = trainXGBoost(data, classes, k_fold, scoring)
    nb = trainNaiveBayes(data, classes, k_fold, scoring)