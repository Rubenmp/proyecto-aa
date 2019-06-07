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


def resample(X, y):
    X_res, y_res = list(X), list(y)
    for x, c in zip(X, y):
        if c == 1:
            for i in range(49):
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
    verbosity = 2 if verbose else 0
    classifier = RandomizedSearchCV(classifier, parameters, n_jobs=-1, cv=3,
                                    scoring=scorer, n_iter=n_iter,
                                    verbose=verbosity)

    classifier.fit(dataset.train_var, dataset.train_output)

    print(f"Mejores parametros ({classifier.best_score_}): "
          f"{classifier.best_params_}")

    return classifier


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


# Classification data
ds = get_aps_dataset(small=False)
ds.preprocess()


"""
# Lectura de modelos
model_names = ["Perceptron", "NeuralNetwork", "AdaBoost", "RandomForest"]
models = load_all_models(model_names)
"""

def tuning():
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

    # {'pct__penalty': 'elasticnet', 'pct__alpha': 0.0001, 'pca__n_components': 0.95}
    start_time = time.time()
    pct_clf = tune_parameters(pct_clf, pct_parameters, ds, scorer,
                              verbose=True, n_iter=30)
    print("--- %s seconds ---" % (time.time() - start_time))
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

    start_time = time.time()
    # {'pca__n_components': 0.95, 'mlp__learning_rate': 'constant',
    # 'mlp__hidden_layer_sizes': (100,), 'mlp__alpha': 0.1, 'mlp__
    # activation': 'relu'}
    nn_clf = tune_parameters(nn_clf, nn_parameters, ds, scorer, verbose=True)
    save_model(nn_clf, 'Neural Network CV')
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

    # # 3x5 = 3566s
    start_time = time.time()
    # {'pca__n_components': 0.8, 'ab__learning_rate': 0.1}
    ab_clf = tune_parameters(ab_clf, ab_parameters, ds, scorer, verbose=True)
    save_model(ab_clf, 'AdaBoost CV')
    print("--- %s seconds ---" % (time.time() - start_time))

    # Random Forest

    rf_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full')),
        ('poly', PolynomialFeatures(2)),
        ('rf', RandomForestClassifier(max_features='sqrt', criterion='gini'))
    ])

    rf_parameters = {
        'pca__n_components': [.80, .90, .95, 1],
        'rf__n_estimators': [10, 40, 160],
    }

    start_time = time.time()
    # {'rf__n_estimators': 160, 'rf__criterion': 'gini', 'pca__n_components': 0.8}
    rf_clf = tune_parameters(rf_clf, rf_parameters, ds, scorer, verbose=True, n_iter=12)
    save_model(rf_clf, 'RandomForest CV')
    print("--- %s seconds ---" % (time.time() - start_time))

    models = {
        "Perceptron": pct_clf,
        "Neural network": nn_clf,
        "AdaBoost": ab_clf,
        "Random Forest": rf_clf
    }

    return models


def train():
    pct_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.95)),
        ('poly', PolynomialFeatures(2)),
        ('pct', Perceptron(penalty='elasticnet', alpha=.0001, max_iter=500,
                           tol=.001, class_weight=ds.WEIGHTS_DIC))
    ])

    start_time = time.time()
    pct_clf.fit(ds.train_var, ds.train_output)
    print("--- %s seconds ---" % (time.time() - start_time))

    train_var_res, train_output_res = resample(ds.train_var, ds.train_output)

    nn_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.95)),
        ('poly', PolynomialFeatures(2)),
        ('mlp', MLPClassifier(solver='adam', max_iter=100, tol=0.001,
                              learning_rate='constant',
                              hidden_layer_sizes=(100,), alpha=0.1,
                              activation='relu'))
    ])

    start_time = time.time()
    nn_clf.fit(train_var_res, train_output_res)
    print("--- %s seconds ---" % (time.time() - start_time))

    ab_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.8)),
        ('poly', PolynomialFeatures(2)),
        ('ab', AdaBoostClassifier(learning_rate=0.1))
    ])

    start_time = time.time()
    ab_clf.fit(train_var_res, train_output_res)
    print("--- %s seconds ---" % (time.time() - start_time))

    rf_clf = Pipeline(steps=[
        ('pca', PCA(svd_solver='full', n_components=0.8)),
        ('poly', PolynomialFeatures(2)),
        ('rf', RandomForestClassifier(max_features='sqrt', n_estimators=160,
                                      criterion='gini',
                                      class_weight=ds.WEIGHTS_DIC))
    ])

    start_time = time.time()
    rf_clf.fit(ds.train_var, ds.train_output)
    print("--- %s seconds ---" % (time.time() - start_time))

    models = {
        "Perceptron": pct_clf,
        "Neural network": nn_clf,
        "AdaBoost": ab_clf,
        "Random Forest": rf_clf
    }

    for model_name in models:
        save_model(models[model_name], model_name)

    return models


def yes_or_no(question):
    while True:
        ans = input(question)
        if len(ans) > 0:
            if ans[0] == "S" or ans[0] == "s":
                ans = True
                break
            elif ans[0] == "N" or ans[0] == "n":
                ans = False
                break
    return ans


def main():
    do_tuning = yes_or_no("¿Desea hacer la estimación de los hiperparámetros "
                          "para cada modelo? (Puede tardar algunas horas) "
                          "[Sí/No] ")

    if do_tuning:
        models = tuning()
    else:
        do_training = yes_or_no("¿Desea hacer el entrenamiento con los "
                                "parámetros que figuran en la memoria de la "
                                "práctica? (Puede tardar unos minutos; "
                                "alternativamente puede leer los modelos ya "
                                "entrenados) [Sí/No] ")
        if do_training:
            models = train()
        else:
            print("Procederemos a leer los modelos ya entrenados.")
            models = load_all_models(("Perceptron", "Neural network",
                                      "AdaBoost", "Random Forest"))

    for model_name in models:
        model = models[model_name]
        score = model.score(ds.test_var, ds.test_output,
                            sample_weight=ds.get_sample_weight_test())
        score_train = model.score(ds.train_var, ds.train_output,
                                  sample_weight=ds.get_sample_weight_train())
        preds = model.predict(ds.test_var)
        print(score_f(ds.test_output, preds))
        print(f"Precisión ponderada del modelo {model_name}: "
              f"{score, score_train}", )


main()
