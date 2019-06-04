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
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight


print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")


def score_f(y_true, y_pred):
    """
        Score function for hyperparameter optimization
    """
    w_dic = DataSet.WEIGHTS_DIC
    sample_weight=compute_sample_weight(w_dic, y_true)
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

# Scorer which takes into account costs of false positives and false negatives
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



def tune_parameters(classifier, parameters, dataset, scorer, n_iter=5, verbose=False):
    verbosity = 2 if verbose else 0
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=3,
                                    scoring=scorer, n_iter=n_iter, verbose=verbosity)
    classifier.fit(dataset.train_var, dataset.train_output)

    y_true, y_pred = dataset.test_output, classifier.predict(dataset.test_var)
    score = score_f(y_true, y_pred)

    print(f"Mejores parametros ({score}): {classifier.best_params_}")

    return classifier, score



# Classification data
ds = get_dataset(small=True)
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
tune_parameters(pct_clf, pct_parameters, ds, scorer, verbose=True)




# Neural network

nn_clf = Pipeline(steps=[('imputer', Imputer()),    
                         ('pca', PCA(svd_solver='full')),
                         ('poly', PolynomialFeatures(2)),
                         ('mlp', MLPClassifier(solver='adam'))])
nn_parameters = {
    'imputer__strat': ['mean', 'median'],
    'pca__n_components': [.80, .90, .95],
    'mlp__hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__alpha': np.logspace(-4, -1, num=5, base=10),
    'mlp__learning_rate': ['constant', 'adaptive']
}

# 2:58
tune_parameters(nn_clf, nn_parameters, ds, scorer, verbose=True)



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
tune_parameters(ab_clf, ab_parameters, ds, scorer, verbose=True)



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
tune_parameters(rf_clf, rf_parameters, ds, scorer, verbose=True)


models = {
    "SVM": pct_clf,
    "Neural network": nn_clf,
    "AdaBoost": ab_clf,
    "Random Forest": rf_clf
}
