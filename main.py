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
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import joblib
import time


##### Functions to save and load models ########################################

def model_file(name):
    return './models/' + name + '.model'


def save_model(model, name):
    joblib.dump(model, open(model_file(name), 'wb'), compress=True)


def load_model(name):
    return joblib.load(open(model_file(name), 'rb'))


def load_all_models(model_names):
    models = {}
    for name in model_names:
        models[name] = load_model(name)

    return models

################################################################################

model_names = ("Perceptron", "Neural network",
                                      "AdaBoost", "Random Forest")

model_names_cv = ("Perceptron CV", "Neural Network CV",
                                      "AdaBoost CV", "Random Forest CV")                                      

########################
#    Score functions
########################
def score_f(y_true, y_pred):
    """
        Score function for hyperparameter optimization
    """
    sample_weight = get_sample_weight(y_true)
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)


# Scorer which takes into account costs of false positives and false negatives
scorer = make_scorer(score_f)



########################
#  Auxiliary functions
########################

def resample(X, y):
    """
    This resampling function returns a modified version of the dataset where
    the elements of the positive class are repeated 50 times.
    """
    X_res, y_res = list(X), list(y)
    for x, c in zip(X, y):
        if c == 1:
            for _ in range(49):
                X_res.append(x)
                y_res.append(-1)
    X_res, y_res = np.array(X_res), np.array(y_res)
    np.random.shuffle(X_res)
    np.random.shuffle(y_res)

    return X_res, y_res


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
    """
    Find the optimal parameters of a model
    """
    verbosity = 2 if verbose else 0
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=3,
                                    scoring=scorer, n_iter=n_iter,
                                    verbose=verbosity)

    classifier.fit(dataset.train_var, dataset.train_output)

    print(f"Mejores parametros ({classifier.best_score_}): "
          f"{classifier.best_params_}")

    return classifier


def tuning(ds):
    """
    Estimación de parámetros
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

    pct_clf = tune_parameters(pct_clf, pct_parameters, ds, scorer,
                              verbose=True, n_iter=30)
    save_model(pct_clf, 'Perceptron CV')

    # Neural network
    nn_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full')),
        ('poly', PolynomialFeatures(2)),
        ('mlp', MLPClassifier(solver='adam', max_iter=100, tol=0.001))
    ])

    nn_parameters = {
        'pca__n_components': [.80, .90, .95, 1],
        'mlp__hidden_layer_sizes': [(100,), (50, 50), (33, 33, 33)],
        'mlp__activation': ['logistic', 'tanh', 'relu'],
        'mlp__alpha': np.logspace(-4, -1, num=4, base=10),
        'mlp__learning_rate': ['constant', 'adaptive']
    }

    nn_clf = tune_parameters(nn_clf, nn_parameters, ds, scorer, verbose=True)
    save_model(nn_clf, 'Neural Network CV')

    # AdaBoost
    ab_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full')),
        ('poly', PolynomialFeatures(2)),
        ('ab', AdaBoostClassifier())
    ])

    ab_parameters = {
        'pca__n_components': [.80, .90, .95, 1],
        'ab__learning_rate': np.logspace(-2, 2, num=5, base=10)
    }

    ab_clf = tune_parameters(ab_clf, ab_parameters, ds, scorer, verbose=True)
    save_model(ab_clf, 'AdaBoost CV')

    # Random Forest
    rf_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full')),
        ('poly', PolynomialFeatures(2)),
        ('rf', RandomForestClassifier(max_features='sqrt', criterion='gini', class_weight=ds.WEIGHTS_DIC))
    ])

    rf_parameters = {
        'pca__n_components': [.80, .90, .95, 1],
        'rf__n_estimators': [10, 40, 160],
        'rf__max_depth' : [15, 25, 50],
    }

    rf_clf = tune_parameters(rf_clf, rf_parameters, ds, scorer, verbose=True, n_iter=12)
    save_model(rf_clf, 'RandomForest CV')

    models = {
        "Perceptron": pct_clf,
        "Neural network": nn_clf,
        "AdaBoost": ab_clf,
        "Random Forest": rf_clf
    }

    return models


def train(ds, save_models=True):
    pct_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.95)),
        ('poly', PolynomialFeatures(2)),
        ('pct', Perceptron(penalty='elasticnet', alpha=.0001, max_iter=500,
                           tol=.001, class_weight=ds.WEIGHTS_DIC))
    ])
    pct_clf.fit(ds.train_var, ds.train_output)     

    # Some models do not allow fitting with weights,
    # training with weights can be simulated copying 
    # the same item several times in order to increase its importance
    train_var_res, train_output_res = resample(ds.train_var, ds.train_output)

    nn_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.95)),
        ('poly', PolynomialFeatures(2)),
        ('mlp', MLPClassifier(solver='adam', max_iter=100, tol=0.001,
                              learning_rate='constant',
                              hidden_layer_sizes=(100,), alpha=0.1,
                              activation='relu'))
    ])
    nn_clf.fit(train_var_res, train_output_res)

    ab_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.8)),
        ('poly', PolynomialFeatures(2)),
        ('ab', AdaBoostClassifier(learning_rate=0.1))
    ])
    ab_clf.fit(train_var_res, train_output_res)

    rf_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.8)),
        ('poly', PolynomialFeatures(2)),
        ('rf', RandomForestClassifier(max_features='sqrt', n_estimators=160,
                                      criterion='gini',
                                      class_weight=ds.WEIGHTS_DIC,
                                      max_depth=25))
    ])
    rf_clf.fit(ds.train_var, ds.train_output)

    models = {
        "Perceptron": pct_clf,
        "Neural network": nn_clf,
        "AdaBoost": ab_clf,
        "Random Forest": rf_clf
    }

    if save_model:
        for model_name in models:
            save_model(models[model_name], model_name)

    return models


########################
#  Results
########################

def truncate_decimals(number):
    return format(number, '.4f')


def print_results_table(train_r, test_r):
    table =  f"""
    +---------------+------------------------+---------------+
    | Modelo        | Score de entrenamiento | Score de test |
    +---------------+------------------------+---------------+
    | Perceptron    |         {train_r[0]}         |   {test_r[0]}      |
    +---------------+------------------------+---------------+
    | Red neuronal  |         {train_r[1]}         |   {test_r[1]}      |
    +---------------+------------------------+---------------+
    | AdaBoost      |         {train_r[2]}         |   {test_r[2]}      |
    +---------------+------------------------+---------------+
    | Random Forest |         {train_r[3]}         |   {test_r[3]}      |
    +---------------+------------------------+---------------+
    """
    print(table)


def show_results(ds, p_models=None):
    models = p_models
    if p_models == None:
        models = load_all_models(model_names)

    train_results = []
    test_results  = []
    for name, model in models.items():
        print(name)
        train_pred = model.predict(ds.train_var)
        train_acc = score_f(ds.train_output, train_pred)
        train_results.append(truncate_decimals(train_acc))
        print(f"Score en training: {train_acc}")

        # Results in test
        test_pred = model.predict(ds.test_var)
        test_acc = score_f(ds.test_output, test_pred)
        test_results.append(truncate_decimals(test_acc))
        print(f"Score en test: {test_acc}")
        print("\n")

    print_results_table(train_results, test_results)


def yes_or_no(question):
    """
    (Auxiliary)
    Function for asking yes/no questions
    """
    while True:
        ans = input(question)
        if len(ans) > 0:
            if ans[0].lower() == "s":
                ans = True
                break
            elif ans[0].lower() == "n":
                ans = False
                break
    return ans


def main():
    print("Problema de clasificación APS Failure at Scania Trucks Data Set\n")

    # Classification data
    ds = get_aps_dataset(small=True)
    ds.preprocess()

    do_tuning = yes_or_no("¿Desea hacer la estimación de los hiperparámetros "
                          "para cada modelo? (Puede tardar algunas horas) "
                          "[Sí/No] ")

    if do_tuning:
        models = tuning(ds)
    else:
        do_training = yes_or_no("¿Desea hacer el entrenamiento con los "
                                "parámetros que figuran en la memoria de la "
                                "práctica? (Puede tardar unos minutos; "
                                "alternativamente puede leer los modelos ya "
                                "entrenados) [Sí/No] ")
        if do_training:
            models = train(ds)
        else:
            print("Procederemos a leer los modelos ya entrenados.")
            models = load_all_models(model_names)

    show_results(ds, p_models=models)


if __name__ == "__main__":
    main()
