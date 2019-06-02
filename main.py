#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight


random_state = 0

print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")


def score_f(y_true, y_pred):
    """
        Score function for hyperparameter optimization
    """
    w_dic = DataSet.weights_dic
    sample_weight=compute_sample_weight(w_dic, y_true)
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

scorer = make_scorer(score_f)


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

        score += model.score(X_test, y_test,
                             dataset.get_sample_weight_train()[test_idx])
        
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


def tune_parameters(classifier, parameters):
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=5,
                                    scoring=scorer, n_iter=1)
    classifier.fit(ds.train_var, ds.train_output)

    y_true, y_pred = ds.test_output, classifier.predict(ds.test_var)
    score = score_f(y_true, y_pred)

    print(score)
    print("mejores parametros: " + str(classifier.best_params_))

    return classifier, score

# Classification data
ds = get_dataset(small=False)
ds.preprocess()


# Neural network

nn_clf = Pipeline(steps=[('pca', PCA(svd_solver='full')),
                         ('poly', PolynomialFeatures()),
                         ('mlp', MLPClassifier(solver='adam'))])
nn_parameters = {
    'pca__n_components': [.80, .90, .95],
    'poly__degree': [1, 2, 3],
    'mlp__hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__alpha': np.logspace(-4, -1, num=5, base=10),
    'mlp__learning_rate': ['constant', 'adaptive']
}

# 2:58
# nn_clf = RandomizedSearchCV(nn_clf, nn_parameters, n_jobs=-1, cv=5, scoring=scorer, n_iter=1)
# nn_clf.fit(ds.train_var, ds.train_output)
#
# y_true, y_pred = ds.test_output, nn_clf.predict(ds.test_var)
# score = score_f(y_true, y_pred)
# print(score)
# print("mejores parametros: " + str(nn_clf.best_params_))


# Linear regression

pct_clf = Pipeline(steps=[('pca', PCA(svd_solver='full')),
                          ('poly', PolynomialFeatures(2)),
                          ('pct', Perceptron(max_iter=10,
                                             tol=.001,
                                             n_jobs=-1))])
pct_parameters = {
    'pca__n_components': [.80, .90, .95],
    'pct__penalty': ['l1', 'l2', 'elasticnet'],
    'pct__alpha': np.logspace(-5, -1, num=5, base=10),
}

# # 2:23
pct_clf = RandomizedSearchCV(pct_clf, pct_parameters,
                             cv=5, scoring=scorer, n_iter=1)

pct_clf.fit(ds.train_var, ds.train_output)

y_true, y_pred = ds.test_output, pct_clf.predict(ds.test_var)
score = score_f(y_true, y_pred)
print(score)
print("mejores parametros: " + str(pct_clf.best_params_))

# tune_parameters(pct_clf, pct_parameters)

# AdaBoost

ab_clf = Pipeline(steps=[('pca', PCA(svd_solver='full')),
                         ('poly', PolynomialFeatures(2)),
                         ('ab', AdaBoostClassifier())]) #TODO: base estimator
                                                        #TODO: SAMME.R vs SAMME

ab_parameters = {
    'pca__n_components': [.80, .90, .95],
    'ab__learning_rate': np.logspace(-2, 2, num=5, base=10)
}

# # 3:22
# ab_clf = RandomizedSearchCV(ab_clf, ab_parameters,
#                             cv=5, scoring=scorer, n_iter=1)
#
# ab_clf.fit(ds.train_var, ds.train_output)
#
# y_true, y_pred = ds.test_output, ab_clf.predict(ds.test_var)
# score = score_f(y_true, y_pred)
# print(score)
# print("mejores parametros: " + str(ab_clf.best_params_))


# Random Forest

rf_clf = Pipeline(steps=[('pca', PCA(svd_solver='full')),
                         ('poly', PolynomialFeatures(2)),
                         ('rf', RandomForestClassifier(

                            max_features='sqrt'
                         ))])

rf_parameters = {
    'pca__n_components': [.80, .90, .95],
    'rf__n_estimators': [10, 40, 160],
    'rf__criterion': ['gini', 'entropy']
}

# 4-5 min. de media
# rf_clf = RandomizedSearchCV(rf_clf, rf_parameters,
#                             cv=5, scoring=scorer, n_iter=1)
#
# rf_clf.fit(ds.train_var, ds.train_output)
#
# y_true, y_pred = ds.test_output, rf_clf.predict(ds.test_var)
# score = score_f(y_true, y_pred)
# print(score)
# print("mejores parametros: " + str(rf_clf.best_params_))

models = {
    "SVM": pct_clf,
    "Neural network": nn_clf,
    "AdaBoost": ab_clf,
    "Random Forest": rf_clf
}