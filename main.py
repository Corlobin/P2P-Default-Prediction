#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:18:01 2018

@author: diegosoaresub
"""
import csv
import itertools
import requests
import pandas as pd
import randomForestUtil as rfu
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime
import threading
from itertools import product
from joblib import Parallel, delayed

def all_subsets(ss):
    return sorted(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

def calculate_feature_accuracy(loans_features):
    X = loans_features.drop("loan_status", axis=1)
    Y = loans_features.loan_status

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=123)
    #y_pred = rfu.random_forest(x_train, x_test, y_train)
    y_pred = rfu.gradient_boosting(x_train, x_test, y_train)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy

pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

loans = pd.read_csv('train_70.csv', low_memory=False)
# loans_test = pd.read_csv('../processed_data/test_30.csv', low_memory=False)
# loans = pd.concat([loans, loans_test])
# loans = pd.read_csv('../processed_data/loan_2007_2016Q4_old.csv', low_memory=False)
# loans = pd.read_csv('../processed_data/train_70.csv', low_memory=False)

# Balanced data

loans_defaulted = loans[loans.loan_status == 0];
loans_ok = loans[loans.loan_status == 1].head(len(loans_defaulted))
loans = pd.concat([loans_ok, loans_defaulted])

print('Samples per class: {}'.format(loans_ok.shape[0]))


selected_features = ['loan_status',
    'grade',
    #'dti',
    'annual_inc',
    #'revol_util',
    'int_rate',
   # 'revol_bal',
   # 'total_acc',
   # 'tot_cur_bal',
    # installment
    'loan_amnt',
#    'pct_tl_nvr_dlq',
#    'mo_sin_rcnt_rev_tl_op',
#    'num_rev_accts',
#    'num_il_tl',
#    'acc_open_past_24mths',
#    'mo_sin_rcnt_tl',
#    'num_bc_tl',
#    'emp_length',
 #   'percent_bc_gt_75',
    #'open_acc'
#    'num_op_rev_tl'
]


#feature_combinations = set(all_subsets(selected_features[1:]))
#feature_combinations = list(all_subsets(selected_features[1:]))

feature_combinations=list([('grade', 'annual_inc', 'int_rate','loan_amnt')])

'''test_feature_combinations = sorted(feature_combinations)
test_set_feature_combinations = list()
already_created_list = dict()
actual = 0
for combine in test_feature_combinations:
    actual = actual  + 1
    if actual % 1000 == 0:
        print('actual : ' + str(actual))
    if len(combine) > 0:
        ordered = sorted(combine)
        dict_key = ''.join(ordered)
        if dict_key not in already_created_list:
            test_set_feature_combinations = test_set_feature_combinations + [ordered]
            already_created_list[dict_key] = True
        else:
            print('already exists ' + dict_key)'''

print('Total combinations: ' + str(len(feature_combinations)))
#print('Total combinations: ' + str(len(test_set_feature_combinations)))

# calculate each accuracy for feature combination
features_accuracy = dict()
actual_combination = 0

for subset in feature_combinations:
    actual_combination = actual_combination + 1
    if (len(subset) > 0):
        ini = datetime.now()
        print('\t Calculing for combination ' + str(actual_combination))
        subset_training = ['loan_status'] + list(subset)  # add loan status
        loans_calculate = loans[subset_training]  # select features
        acc = calculate_feature_accuracy(loans_calculate)
        feature_index_list = feature_combinations.index(subset)
        features_accuracy[feature_index_list] = acc
        end = datetime.now()
        diff = end - ini
        print('\t\t Time for execution ' + str(diff))

w = csv.writer(open("1_gradient_boosting_output1.csv", "w"))
for key, val in features_accuracy.items():
    w.writerow([feature_combinations[key], val])

print('End analysis')


