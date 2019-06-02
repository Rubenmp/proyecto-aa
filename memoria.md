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

Otra características que cabe señalar es la cantidad de atributos que faltan (al 99% de los datos le falta algún atributo), que toda las variables están anonimizadas por motivos corporativos y que la distribución de clases está muy desbalanceada: solo el 1,7% de los ejemplos del conjunto de entrenamiento pertenecen a la clase positiva.

Trataremos de diseñar un modelo para este problema de cada uno de los siguientes tipos:

* SVM (Support Vector Machine);
* Red neuronal;
* Random Forest;
* AdaBoost.

# Preprocesado de datos

## Tratamiento de valores desconocidos

El principal problema a resolver en el preprocesado de datos es la cantidad de valores desconocidos en el conjunto.

El 99% de los ejemplos tienen algún valor desconocido, de modo que no es viable eliminar los ejemplos con valores desconocidos.

Es razonable plantearse la posibilidad de eliminar variables con valores desconocidos para la mayoría de ejemplos. Sin embargo, podemos ver en la Figura 1 que la amplia mayoría de variables tienen menos de un 10% de valores desconocidos y ninguna supera el rango de 80% de valores desconocidos. Por tanto eliminar variables supondría sacrificar demasiada información.

![](./imgs/histograma_nan.png)

Optamos por una estrategia de imputación de valores desconocidos. Esta estrategia consiste en asignar valores donde falten en función del resto de valores del conjunto de datos. Esto se puede hacer calculando los valores en función de los valores de la misma variable para el resto de ejemplos (imputación univariante) o en función de los valores del resto de variables de ese ejemplo (imputación multivariante). La imputación univariante se hace con algún estadístico de los valores de cada variable, como la media, la moda o la mediana.

TODO: discutir cuál y por qué

## Outliers


## Normalización de datos


## Análisis de componentes principales

Existen diferentes técnicas para reducir la dimensionalidad de los datos. Se podrían eliminar las variables con alta correlación o varianza muy baja. Otra técnica más adecuada es PCA (Principal Component Analysis), que considera nuevas variables (combinaciones lineales de las originales) no correlacionadas llamadas componentes principales, ordenadas por la cantidad de varianza original que describen. Posteriormente, se elige el porcentaje de varianza que se quiere poder explicar y se eliminan las componentes principales que no sean relevantes, por ello se trata de una técnica de reducción de la dimensionalidad.

Una ventaja de esta técnica es que ayuda a reducir el sobreajuste, al quedarse con la información esencial del problema. La desventaja de este análisis es que de que se pierde interpretabilidad de los datos al optimizar con combinaciones lineales de los mismos. En este caso no nos importa esta desventaja ya que las variables iniciales estaban ya anonimizadas.


# Selección de clases de funciones

TODO: decidir cuáles (comentar que la anonimización de las variables hace que vayamos a ciegas)

# Definición de los conjuntos de training y test

Los conjuntos de training y test vienen ya dados. El conjunto de training consta de 60.000 ejemplos, 59.000 de la clase negativa y 1.000 de la clase positiva. El conjunto de test tiene 16.000 ejemplos.

Para la validación dividiremos el conjunto de training en cinco subconjuntos disjuntos, rotando el subconjunto que se usa como validación y usando los otros cuatro como entrenamiento (validación cruzada).

# Regularización

# Definición de los modelos y estimación de parámetros


Cada modelo tiene diferentes parámetros que pueden ajustarse para mejorar la eficacia para cada conjunto de datos. Por ejemplo, SVM tiene diferentes funciones kernel (lineal, polinómica, rbf, sigmoide). Hay algunos parámetros que pueden elegirse mediante el estudio del problema, si se ve claramente que los datos son linealmente separables no se necesita probar con todos los kernels. Otras veces la elección no está tan clara, por ello se puede llevar a cabo optimización de hiperparámetros. En esta técnica se elige para cada hiperparámetro un conjunto de valores con los que probar y se busca la mejor combinación de estos mediante validación cruzada dentro del conjunto de entrenamiento. Si hay muchos hiperparámetros que ajustar puede que por tiempo de cómputo no interese probar con todas las combinaciones, por ello existen estimadores de hiperparámetros que hacen búsquedas aleatorizadas.


## SVM

## Red Neuronal

## Boosting

## Random Forest

# Métrica del ajuste

En este problema se especifica cuál que el objetivo es minimizar el coste, definido como

$$\text{coste\_total} = \text{coste\_1} \times \text{FP} + \text{coste\_2} \times \text{FN} \text{,}$$

donde $\text{coste\_1} = 10$, $\text{coste\_2} = 500$ y $\text{FP}$ y $\text{FN}$ denotan, respectivamente, el número de datos incorrectamente clasificados por el modelo como positivos y negativos. Es decir, el coste de un falso negativo (no detectar la verdadera causa de la avería) es mucho mayor que el de un falso positivo (arreglar innecesariamente el APS).

Por tanto, una métrica de la bondad del ajuste tiene que cumplir que su maximización sea equivalente a la minimización de $\text{coste\_total}$. Una posibilidad es usar una tasa de acierto ponderada de la siguiente manera:

$$\text{tasa\_acierto\_ponderada} = \frac{50 \times \text{VP} + \text{VN}}{50 \times \text{P} + \text{N}} \text{,}$$

donde $P$ y $N$ denotan el número de ejemplos datos positivos y negativos y $VP$ y $VN$ representan, respectivamente, el número de datos correctamente clasificados por el modelo como positivos y negativos.

Podemos comprobar que minimizar $\text{coste\_total}$ es equivalente a maximizar $\text{tasa\_acierto\_ponderada}$. En efecto, maximizar $\text{tasa\_acierto\_ponderada}$ equivale a maximizar $50 \times \text{VP} + \text{VN} = 50 \times \text{P} - 50 \times \text{FP} + \text{N} - \text{FN}$ porque $50 \times \text{P} + \text{N}$ es constante, y por la misma razón es equivalente a minimizar $50 \times \text{FP} + \text{FN}$, que es obviamente lo mismo que minimizar $\text{coste\_total}$.

La métrica $\text{tasa\_acierto\_ponderada}$ tiene la virtud de estar acotada entre 0, que representa que el coste es el máximo posible, y 1, que representa que el coste es el mínimo posible.

TODO: terminar




# Estimación del error

# Justificación de la calidad del modelo
