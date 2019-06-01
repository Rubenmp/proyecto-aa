#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *
from sklearn.model_selection import KFold
import sklearn

import pandas as pd

random_state = 0

print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")


def validate(dataset, model):
    X = dataset.train_var.values
    y = dataset.train_output

    n_splits = 5
    kf = KFold(n_splits=n_splits)

    score = 0
    for train_idx, test_idx in kf.split(X):
        # train_idx and test_idx are indexes over
        # training examples in dataset, test_idx represents
        # indexes of validation set
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        score += model.score(X_test, y_test, dataset.get_sample_weight_train()[test_idx])
        
    return score/n_splits


def best_model_val(dataset, models):
    scores = []
    for model in models:
        score = validate(dataset, model)
        scores.append(score)

    best_model_idx = 0
    return models[best_model_idx]


def best_svm(dataset):
    svm_clf1 = sklearn.svm.SVC(kernel='linear', gamma='scale')
    svm_clf2 = sklearn.svm.SVC(kernel='poly', gamma='scale')
    models = [svm_clf1, svm_clf2]

    return best_model_val(dataset, models)


# Classification data
ds = get_dataset(small=True)
ds.preprocess()




# SVM
svm_clf = best_svm(ds)



# Neural nets


# Boosting


# Random forest