#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO
Autores:
- Rubén Morales Pérez
- Francisco Javier Morales Piqueras
"""

from dataset import *
from imputer import *
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
        score = cross_val_score(models[model], dataset.train_var, dataset.train_output, scoring=scorer, cv=5, n_jobs=-1)
        scores[model] = score
        if score > max_score:
            max_score = score
            max_score_model = model

    for model in models:
        print(f"El score de {model} es {scores[model]}.")

    print(f"El mejor modelo es {max_score_model}.")

    return max_score_model, models[max_score_model]



def tune_parameters(classifier, parameters, dataset, scorer, n_iter=5, verbose=False):
    verbosity = 2 if verbose else 0
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=5,
                                    scoring=scorer, n_iter=n_iter, verbose=verbosity)
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


# Classification data
ds = get_aps_dataset(small=True)
ds.preprocess()



# Perceptrón

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
pct_clf, _ = tune_parameters(pct_clf, pct_parameters, ds, scorer,verbose=True)
save_model(pct_clf, 'Perceptron')



# Neural network

nn_clf = Pipeline(steps=[('pca', PCA(svd_solver='full')),
                         ('poly', PolynomialFeatures(2)),
                         ('mlp', MLPClassifier(solver='adam'))])
nn_parameters = {
    'pca__n_components': [.80, .90, .95],
    'mlp__hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__alpha': np.logspace(-4, -1, num=5, base=10),
    'mlp__learning_rate': ['constant', 'adaptive']
}

# 2:58
nn_clf, _ = tune_parameters(nn_clf, nn_parameters, ds, scorer, verbose=True)
save_model(nn_clf, 'NeuralNet')


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
ab_clf, _ = tune_parameters(ab_clf, ab_parameters, ds, scorer, verbose=True)
save_model(ab_clf, 'AdaBoost')



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
rf_clf, _ = tune_parameters(rf_clf, rf_parameters, ds, scorer, verbose=True)
save_model(rf_clf, 'RandomForest')


models = {
    "SVM": pct_clf,
    "Neural network": nn_clf,
    "AdaBoost": ab_clf,
    "Random Forest": rf_clf
}
