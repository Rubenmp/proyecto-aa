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

En este trabajo trataremos el conjunto de datos *APS failure at Scania Trucks*. Este conjunto de datos contiene información sobre el funcionamiento del sistema de presión del aire (*APS*) de camiones Scania, que se encarga de funciones como el frenado. 
Se recogen 171 variables seleccionadas por expertos en casos de averías y se asigna a cada dato una clase *positiva* (la avería está relacionada con el APS) o *negativa* (la avería no está relacionada con el APS). El conjunto de entrenamiento proporcionado tiene 60000 ejemplos y el de test tiene 16000.

El problema consiste en predecir la clase de un ejemplo a partir de los atributos, por tanto es un problema de **clasificación**. 
Según se especifica en la descripción del conjunto de datos, no todos los errores de clasificación tienen el mismo coste: el coste de los falsos positivos es 50 veces mayor que el de los falsos negativos.

Otra características que cabe señalar es la cantidad de atributos que faltan (al 99% de los datos le falta algún atributo), que toda las variables están anonimizadas por motivos corporativos y que la distribución de clases está muy desbalanceada: solo el 1,7% de los ejemplos del conjunto de entrenamiento pertenecen a la clase positiva.

Trataremos de diseñar un modelo para este problema de cada uno de los siguientes tipos:

* Perceptrón;
* Red neuronal;
* Random forest;
* AdaBoost.

# Preprocesado de datos

## Tratamiento de valores desconocidos

El principal problema a resolver en el preprocesado de datos es la cantidad de valores desconocidos en el conjunto.
El 99% de los ejemplos tienen algún valor desconocido, de modo que no es viable eliminar directamente los ejemplos con valores desconocidos.

Es razonable plantearse la posibilidad de eliminar variables con valores desconocidos para la mayoría de ejemplos. Sin embargo, podemos ver en la Figura 1 que la amplia mayoría de variables tienen menos de un 10% de valores desconocidos y ninguna supera el rango de 80% de valores desconocidos. Por tanto eliminar variables supondría sacrificar demasiada información.

![](./imgs/histograma_nan.png){ width=85% }

Optamos por una estrategia de imputación de valores desconocidos. 
Esta estrategia consiste en asignar valores donde falten en función del resto de valores del conjunto de datos. 
Esto se puede hacer calculando los valores en función de los valores de la misma variable para el resto de ejemplos (imputación univariante) o en función de los valores del resto de variables de ese ejemplo (imputación multivariante). 
La imputación univariante se hace con algún estadístico de los valores de cada variable, como la media, la moda o la mediana.

TODO: discutir cuál y por qué


## Tratamiento de outliers

Los datos a veces pueden tener errores, por ello es deseable hacer un tratamiento previo de los datos problemáticos. 
Hay dos tipos principales de análisis de outliers, univariable y multivariable.

Cuando se tratan los outliers en problemas de clasificación hay que tener cuidado, no se deben aplicar directamente las técnicas de detección de outliers sin tener en cuenta la clase de cada elemento.
Si se hace el tratamiento de outliers ignorando la clase en un conjunto muy desbalanceado se corre el riesgo de que la mayoría de elementos de la clase minoritaria sean clasificados como outliers. 
En nuestro conjunto de entrenamiento si se aplica una detección de outliers sin considerar la clase obtenemos $2785$ outliers (de un total de $60.000$ elementos), $859$ de ellos tienen clase positiva (de un total de 1.000 casos positivos), estaríamos eliminando la mayoría de elementos de la clase positiva. 
Por este motivo es deseable buscar los outliers entre los casos positivos y luego entre los negativos de forma independiente.

TODO: citar justificación separar tratamiento de outliers por clases
https://idus.us.es/xmlui/bitstream/handle/11441/42708/Deletin%20or%20keeping.pdf?sequence=1&isAllowed=y

### Isolation Forests
El algoritmo usado para la detección de outliers es Isolation Forests.

TODO: explicar algoritmo


![](./imgs/boxplot_aa_000.png){ width=50% } ![](./imgs/new_boxplot_aa_000.png){ width=50% }

En el primer gráfico se han calculado los candidatos a outliers respecto a una variable, todo esto dentro del conjunto de entrenamiento y con la clase positiva, aplicar detección de outliers sobre el test no tiene sentido. 
Se ha realizado el boxplot sin tener en cuenta dichos candidatos y se han añadido después como rombos. 
En la segunda imagen se han eliminado los outliers usando Isolation Forests multivariable.
Teniendo en cuenta esta aproximación puede aumentar la calidad del conjunto de entrenamiento.



## Normalización de datos


## Análisis de componentes principales

Existen diferentes técnicas para reducir la dimensionalidad de los datos. Se podrían eliminar las variables con alta correlación o varianza muy baja. 
Otra técnica más adecuada es PCA (Principal Component Analysis), que considera nuevas variables (combinaciones lineales de las originales) no correlacionadas llamadas componentes principales, ordenadas por la cantidad de varianza original que describen. 
Posteriormente, se elige el porcentaje de varianza que se quiere poder explicar y se eliminan las componentes principales que no sean relevantes, por ello se trata de una técnica de reducción de la dimensionalidad.

Una ventaja de esta técnica es que ayuda a reducir el sobreajuste, al quedarse con la información esencial del problema. 
La desventaja de este análisis es que de que se pierde interpretabilidad de los datos al optimizar con combinaciones lineales de los mismos. 
En este caso no nos importa esta desventaja ya que las variables iniciales estaban ya anonimizadas. 
No existe un porcentaje de varianza perfecto con el que determinar el número de componentes principales con el que debemos quedarnos, por ello se probará con diferentes valores en validación cruzada estratificada.


# Selección de clases de funciones

TODO: decidir cuáles (comentar que la anonimización de las variables hace que vayamos a ciegas)

# Definición de los conjuntos de training y test

Los conjuntos de training y test vienen ya dados. El conjunto de training consta de 60.000 ejemplos, 59.000 de la clase negativa y 1.000 de la clase positiva. El conjunto de test tiene 16.000 ejemplos.

Para la validación dividiremos el conjunto de training en cinco subconjuntos disjuntos, rotando el subconjunto que se usa como validación y usando los otros cuatro como entrenamiento (validación cruzada).

# Regularización

# Definición de los modelos y estimación de parámetros


Cada modelo tiene diferentes parámetros que pueden ajustarse para mejorar su precisión. Por otro lado hay parámetros que puede ser deseable que cambien en función del modelo pero no ser del propio modelo, por ejemplo, el porcentaje de varianza con el que queremos quedarnos tras aplicar análisis de componentes principales.

Hay parámetros que pueden elegirse mediante el estudio del problema. 
Sin embargo, a veces la elección no está tan clara, por ello se puede llevar a cabo optimización de hiperparámetros. 
En esta técnica se elige para cada hiperparámetro un conjunto de valores con los que probar y se busca la mejor combinación de estos mediante validación cruzada dentro del conjunto de entrenamiento. 
Si hay muchos hiperparámetros que ajustar puede que por tiempo de cómputo no interese probar con todas las combinaciones, por ello existen estimadores de hiperparámetros que hacen búsquedas aleatorizadas.


## Perceptrón
El algoritmo lineal elegido es el perceptrón, una simulación matemática de una neurona. 
El perceptrón tiene unas entradas con unos pesos asociados, combina dichas entradas y después aplica una función de activación. Esto genera una salida que puede ser única, como tendremos en este caso, o múltiple, en un caso más general.

El perceptrón aprende a mejorar su precisión ajustando los pesos con la propagación de errores hacia atrás. Para cada ejemplo se calcula la salida que tendría y se modifican los pesos para obtener una salida más parecida a la real. El problema del perceptrón es que si los datos no son separables mediante un hiperplano su precisión no será demasiado buena.


## Red neuronal
La combinación de perceptrones permite superar la barrera de los discriminadores lineales.
Una red neuronal consiste en una serie de capas de neuronas, cada una conectada con las capas adyacentes. Al igual que en el perceptrón la primera capa representa la entrada, la última capa la salida y el objetivo último es ajustar los pesos de la red.

El teorema de aproximación universal afirma que una red neuronal con una capa oculta puede aproximar cualquier función continua con entradas dentro de un determinado rango. 
En problemas reales puede haber ruido que nos impida una aproximación perfecta, además de otros factores como número de datos insuficiente.

## Boosting
Boosting es un algoritmo que permite combinar algoritmos para obtener mejores resultados.
A cada uno de los algoritmos débiles se le asigna un peso en función de la precisión que tenga, de esta forma mejores algoritmos tendrán más peso en el resultado final, pero no lo determinarán completamente.

El algoritmo planteado es AdaBoost (Adaptive Boosting) una técnica de boosting adaptativa.
La adapción consiste en ir cambiando durante el entrenamiento el peso de los ejemplos, dando mayor importancia a aquellos que fueron clasificados incorrectamente en una iteración.
Una de las críticas principales a este algoritmo es que tiene dificultades para resistir el ruido de la clasificación aleatoria.


## Random forest
Random forest es una mejora sobre los árboles de decisión. Se construyen varios árboles de decisión, cada uno usa un subconjunto aleatorio de las variables explicativas para entrenar un arbol sobre un subconjunto aleatorio de datos. Usar diferentes variables para cada árbol ayuda a que los árboles no estén altamente correlados. Los árboles generados tienen ruido, por ello se hace un promedio de los resultados para elegir la salida final, reduciendo la varianza.

Una diferencia con los árboles de decisión es que la clasificación hecha por random forests es difícil de interpretar, pero esto no es un problema en este caso ya que las variables están inicialmente anonimizadas.


# Métrica del ajuste

En este problema se especifica que el objetivo es minimizar el coste, definido como
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
