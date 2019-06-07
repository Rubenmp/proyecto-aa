#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *
#from imputer import *
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.externals import joblib
import time


print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")


def score_f(y_true, y_pred):
    """
        Score function for hyperparameter optimization
    """
    sample_weight = get_sample_weight(y_true)
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)


# Scorer which takes into account costs of false positives and false negatives
scorer = make_scorer(score_f)


def compare_models(dataset, models):
    scores = {}
    max_score = 0
    max_score_model = None

    for model in models:
        score = cross_val_score(models[model], dataset.train_var,
                                dataset.train_output, scoring=scorer, cv=5,
                                n_jobs=-1)
        scores[model] = score
        if score > max_score:
            max_score = score
            max_score_model = model

    for model in models:
        print(f"El score de {model} es {scores[model]}.")

    print(f"El mejor modelo es {max_score_model}.")

    return max_score_model, models[max_score_model]


def tune_parameters(classifier, parameters, dataset, scorer, n_iter=10,
                    verbose=False):
    verbosity = 2 if verbose else 0
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=3,
                                    scoring=scorer, n_iter=n_iter,
                                    verbose=verbosity)

    classifier.fit(dataset.train_var, dataset.train_output)

    y_true, y_pred = dataset.test_output, classifier.predict(dataset.test_var)
    score = score_f(y_true, y_pred)

    print(f"Mejores parametros ({score}): {classifier.best_params_}")

    return classifier, score


def model_file(name):
    return './models/' + name + '.model'


def save_model(model, name):
    joblib.dump(model, open(model_file(name), 'wb'), compress=True)


def load_model(name):
    return joblib.load(open(model_file(name), 'rb'))


def load_all_models(model_names):
    models = []
    for name in model_names:
        models.append(load_model(name))

    return models


# Classification data
ds = get_aps_dataset(small=True)
ds.preprocess()

"""
# Lectura de modelos
model_names = ["Perceptron", "NeuralNet", "AdaBoost", "RandomForest"]
models = load_all_models(model_names)
"""



# Perceptron

pct_clf = Pipeline(steps=[
    ('pca', PCA(svd_solver='full')),
    ('poly', PolynomialFeatures(2)),
    ('pct', Perceptron(max_iter=500, tol=.001, n_jobs=-1))
])

pct_parameters = {
    'pca__n_components': [.80, .90, .95, 1],
    'pct__penalty': ['l1', 'l2', 'elasticnet'],
    'pct__alpha': np.logspace(-5, -1, num=5, base=10),
}

# 3x20 = 398s
# start_time = time.time()
# pct_clf, score = tune_parameters(pct_clf, pct_parameters, ds, scorer,
#                                  verbose=True, n_iter=50)
# print("--- %s seconds ---" % (time.time() - start_time))
# save_model(pct_clf, 'Perceptron')


# Neural network

nn_clf = Pipeline(steps=[
    ('pca', PCA(svd_solver='full')),
    ('poly', PolynomialFeatures(2)),
    ('mlp', MLPClassifier(solver='adam', max_iter=100, tol=0.001))
])

nn_parameters = {
    'pca__n_components': [.80, .90, .95, 1],
    'mlp__hidden_layer_sizes': [(100,), (50, 50), (33, 33, 33)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__alpha': np.logspace(-4, -1, num=4, base=10),
    'mlp__learning_rate': ['constant', 'adaptive']
}

# 3x10 = 3219s
# start_time = time.time()
# # {'pca__n_components': 0.95, 'mlp__learning_rate': 'adaptive',
# # 'mlp__hidden_layer_sizes': (33, 33, 33), 'mlp__alpha': 0.001,
# # 'mlp__activation': 'relu'}
# nn_clf, score = tune_parameters(nn_clf, nn_parameters, ds, scorer, verbose=True)
# save_model(nn_clf, 'NeuralNet')
# print("--- %s seconds ---" % (time.time() - start_time))


# AdaBoost

ab_clf = Pipeline(steps=[
    ('pca', PCA(svd_solver='full')),
    ('poly', PolynomialFeatures(2)),
    ('ab', AdaBoostClassifier())
])  #TODO: explicar SAMME.R

ab_parameters = {
    'pca__n_components': [.80, .90, .95, 1],
    'ab__learning_rate': np.logspace(-2, 2, num=5, base=10)
}

# 3x5 = 3566s
# start_time = time.time()
# # {'pca__n_components': 0.8, 'ab__learning_rate': 0.1}
# ab_clf, score = tune_parameters(ab_clf, ab_parameters, ds, scorer, verbose=True)
# save_model(ab_clf, 'AdaBoost')
# print("--- %s seconds ---" % (time.time() - start_time))

# Random Forest

rf_clf = Pipeline(steps=[
    ('pca', PCA(svd_solver='full')),
    ('poly', PolynomialFeatures(2)),
    ('rf', RandomForestClassifier(max_features='sqrt'))
])

rf_parameters = {
    'pca__n_components': [.80, .90, .95, 1],
    'rf__n_estimators': [10, 40, 160],
    'rf__criterion': ['gini', 'entropy']
}

# 4-5 min. de media
start_time = time.time()
rf_clf, _ = tune_parameters(rf_clf, rf_parameters, ds, scorer, verbose=True, n_iter=5)
save_model(rf_clf, 'RandomForest')

models = {
    "Perceptron": pct_clf,
    "Neural network": nn_clf,
    "AdaBoost": ab_clf,
    "Random Forest": rf_clf
}
