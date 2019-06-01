#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer
import sklearn

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


def compare_models(dataset, models):
    scores = {}
    max_score = 0
    max_score_model = None

    for model in models:
        score = validate(dataset, models[model])
        scores[model] = score
        if score > max_score:
            max_score = score
            max_score_model = model

    for model in models:
        print(f"El score de {model} es {scores[model]}.")

    print(f"El mejor modelo es {max_score_model}.")

    return max_score_model, models[max_score_model]

svm_models = {
    "SVM con kernel lineal": sklearn.svm.SVC(kernel='linear', gamma='scale'),
    "SVM con kernel polinómico": sklearn.svm.SVC(kernel='poly', gamma='scale')
}


# Classification data
ds = get_dataset(small=True)
ds.preprocess()

#compare_models(ds, svm_models)


# Neural nets


# Boosting


# Random forest






nn_clf = MLPClassifier(max_iter=100)
parameter_space = {
    #'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    #'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    #'learning_rate': ['constant','adaptive'],
}

scorer = make_scorer(accuracy_score)

clf = GridSearchCV(nn_clf, parameter_space, n_jobs=-1, cv=3, scoring=scorer)
clf.fit(ds.train_var, ds.train_output)

# Ver score
y_true, y_pred = ds.test_output , clf.predict(ds.test_var)
print(clf.best_params_)
