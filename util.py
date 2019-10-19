#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:15:36 2017

@author: diegosoaresub
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import metrics


def getLoans(file_name):
    loans = pd.read_csv(file_name, low_memory=False)

    loans['delinq_2yrs'] = loans['delinq_2yrs'].apply(lambda x: 2 if x >= 2 else x)
    loans['inq_last_6mths'] = loans['inq_last_6mths'].apply(lambda x: 3 if x >= 3 else x)

    loans = loans[[
        'loan_status',
        'int_rate',
        'tot_cur_bal',
        'grade',
        'dti',
        'revol_bal',
        'revol_util',
        'annual_inc',
        'loan_amnt',
        'total_acc'
    ]];

    # Balanced data
    loans_defaulted = loans[loans.loan_status == 0];
    loans_ok = loans[loans.loan_status == 1].head(len(loans_defaulted))
    loans = pd.concat([loans_ok, loans_defaulted])

    X = loans.drop('loan_status', axis=1)
    Y = loans.loan_status

    return X, Y


def load_data(file_name):
    loans = pd.read_csv('processed_data/cleaned_loan_without_late16_30.csv', low_memory=False)

    loans = loans[[
        'loan_status',
        'int_rate',
        'dti',
        'revol_util',
        'annual_inc',
        'revol_bal',
        'installment',
        'total_acc',
        'open_acc',
        'loan_amnt',
        'grade'
        # 'emp_length'
        # 'inq_last_6mths',
        # 'delinq_2yrs',
        # 'pub_rec',
        # 'verification_status_Source Verified',
        # 'verification_status_Verified',
        # 'purpose_credit_card',
        # 'purpose_debt_consolidation',
        # 'home_ownership_RENT',
        # 'verification_status_Not Verified',
        # 'home_ownership_MORTGAGE',
        # 'home_ownership_OWN',
        # 'purpose_other',
        # 'term_ 36 months',
        # 'term_ 60 months',
        # 'purpose_home_improvement',
        # 'purpose_small_business',
        # 'purpose_major_purchase',
        # 'purpose_medical',
        # 'purpose_wedding',
        # 'purpose_car',
        # 'purpose_moving',
        # 'purpose_vacation',
        # 'purpose_house',
        # 'collections_12_mths_ex_med',
        # 'purpose_renewable_energy',
        # 'purpose_educational',
        # 'home_ownership_OTHER',
        # 'acc_now_delinq',
        # 'home_ownership_NONE',
        # 'home_ownership_ANY',
        # 'application_type_JOINT',
        # 'application_type_INDIVIDUAL'
    ]];

    # Balanced data
    loans_defaulted = loans[loans.loan_status == 0];
    loans_ok = loans[loans.loan_status == 1].head(len(loans_defaulted))
    loans = pd.concat([loans_ok, loans_defaulted])

    X = loans.drop("loan_status", axis=1)
    Y = loans.loan_status

    loans = loans.drop("loan_status", axis=1)

    # random_forest_cross(X, Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

    return loans, X, Y, x_train, x_test, y_train, y_test


def plot_roc(y_pred, y_true):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    print('AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    #    plt.xlim([-0.1,1.2])
    #    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def get_roc_results(y_true, results):
    plt.title('Receiver Operating Characteristic')

    for algo, y_pred in results.items():
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label='{} = {:0.2f}'.format(algo, roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')


def confusion_matrix_report(classes, y_pred, y_true):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    # Plot non-normalized confusion matrix
    #    plt.figure()
    #    plot_confusion_matrix(cnf_matrix, classes=classes,
    #                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #    plt.figure()
    #    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
    #                          title='Normalized confusion matrix')
    #
    #    plt.show()

    # Classification report
    classify_report = classification_report(y_true=y_true, y_pred=y_pred)
    print(classify_report)

    print("Accuracy: ", metrics.accuracy_score(y_true, y_pred))
    return metrics.accuracy_score(y_true, y_pred)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')