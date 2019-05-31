---
title: Proyecto Final de Aprendizaje Automático
subtitle: APS Failure at Scania Trucks
author: |
    Francisco Javier Morales Piqueras \
    Rubén Morales Pérez
header-includes : |
    \usepackage{ stmaryrd }
toc: true
lang: es
---

\newpage

# Exposición del problema

En este trabajo trataremos el conjunto de datos *APS failure at Scania Trucks*. Este conjunto de datos contiene información sobre el funcionamiento del sistema de presión del aire (*APS*) de camiones Scania, que se encarga de funciones como el frenado. Se recogen 171 variables seleccionadas por expertos en casos de averías y se asigna a cada dato una clase *positiva* (la avería está relacionada con el APS) o *negativa* (la avería no está relacionada con el APS). El conjunto de entrenamiento proporcionado tiene 60000 ejemplos y el de test tiene 16000.

El problema consiste en predecir la clase de un ejemplo a partir de los atributos, por tanto es un problema de **clasificación**. Según se especifica en la descripción del conjunto de datos, no todos los errores de clasificación tienen el mismo coste: el coste de los falsos positivos es 50 veces mayor que el de los falsos negativos.

Otra características que cabe señalar es la cantidad de atributos que faltan (al TODO: ? % de los datos le falta algún atributo) y que toda las variables están anonimizadas por motivos corporativos.

Trataremos de diseñar un modelo para este problema de cada uno de los siguientes tipos:

* SVM (Support Vector Machine)
* Red neuronal
* Random Forest

# Preprocesado de datos

## Tratamiento de valores desconocidos

## Normalización de datos

# Selección de clases de funciones

TODO: decidir cuáles (comentar que la anonimización de las variables hace que vayamos a ciegas)

# Definición de los conjuntos de training y test

Los conjuntos de training y test vienen ya dados. El conjunto de training consta de 60.000 ejemplos, 59.000 de la clase negativa y 1.000 de la clase positiva. El conjunto de test tiene 16.000 ejemplos.

Para la validación dividiremos el conjunto de training en cinco subconjuntos disjuntos, rotando el subconjunto que se usa como validación y usando los otros cuatro como entrenamiento (validación cruzada).

# Regularización

# Definición de los modelos y estimación de parámetros

## SVM

## Red Neuronal

## Boosting

## Random Forest

# Métrica del ajuste

TODO: discutir métrica

# Estimación del error

# Justificación de la calidad del modelo
