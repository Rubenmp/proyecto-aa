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


print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")


def validate(dataset, model, error_func):
    # TODO: error_func
    X = dataset.train_var
    y = dataset.train_output

    kf = KFold(n_splits=5)

    acc_error = 0
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        acc_error += error_func(y_test, model.predict(X_test))


# Clasification data
ds = get_dataset(small=True)
ds.preprocess()


# SVM


# Neural nets


# Boosting


# Random forest
