# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:26:33 2018

@author: diego.rodrigues
"""
import numpy as np
import matplotlib.pyplot as plt
import util

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error


def random_forest(x_train, x_test, y_train):
    "Random Forest"

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    rf = RandomForestClassifier(n_estimators=200,
                                n_jobs=10,
                                #                                random_state=0,
                                verbose=0)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    rf.fit(x_train, y_train)

    # joblib.dump(rf, 'random_forest_model.pkl')

    return rf.predict(x_test)


def random_forest_cross(X, Y, fold):
    clf = RandomForestClassifier(n_estimators=200,
                                 n_jobs=30,
                                 random_state=0,
                                 verbose=0)

    ypred = cross_val_predict(clf, X, Y, cv=fold)

    print("Mean Squared Error: ", mean_squared_error(y_true=Y, y_pred=ypred))

    util.plot_roc(ypred, Y)

    util.confusion_matrix_report(['Default', 'Good'], ypred, Y)
    return {'Floresta Aleatória': ypred}


def gradient_boosting(x_train, x_test, y_train):
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = GradientBoostingClassifier(n_estimators=200,
                                     random_state=0, max_depth=10)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    clf.fit(x_train, y_train)

    #    joblib.dump(clf, 'gradient_boosting_model.pkl')

    return clf.predict(x_test)


def gradient_boosting_cross(X, Y, fold):
    clf = GradientBoostingClassifier(n_estimators=200,
                                     random_state=0, max_depth=15)
    ypred = cross_val_predict(clf, X, Y, cv=fold)

    print("Mean Squared Error: ", mean_squared_error(y_true=Y, y_pred=ypred))

    util.plot_roc(ypred, Y)
    util.confusion_matrix_report(['Default', 'Good'], ypred, Y)

    return {'Gradient Boosting': ypred}


def features_importance(X, Y, names):
    "Gets the feature importance of each attribute"

    rf = RandomForestRegressor()
    rf.fit(X, Y)

    print("Features sorted by their score:")
    importances = sorted(zip(map(lambda x: round(x, 4) * 100, rf.feature_importances_), names),
                         reverse=True)
    names = []
    for importance, name in importances:
        print("\t{0};{1:.4f}".format(name, importance))
        names.append(name)

    names = reversed(names)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)

    # Plot the feature importances of the forest
    plt.figure(figsize=(14, 12))
    # plt.title("Importância dos Atributos")
    plt.barh(range(X.shape[1]), importances[indices],
             color="g", xerr=std[indices], align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(X.shape[1]), names)
    plt.ylim([-1, X.shape[1]])
    plt.xlabel('Relevância')
    #    plt.ylabel('Atributo')
    plt.show()