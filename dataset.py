#############
#  DATASET  #
#############

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
import sklearn


class DataSet:
    pos_class_weight = 500
    neg_class_weight = 10
    weights_dic = {1: pos_class_weight, -1: neg_class_weight}
    
    def __init__(self, train_f, test_f):
        """
            Initialize dataset from file(s)
        """
        self.train_var, self.train_output = None, None
        self.test_var, self.test_output = None, None

        self.read_train_test(train_f, test_f)


    def read_train_test(self, train_f, test_f):
        """
            Read training and test data from files
        """
        self.train_var, self.train_output = self.read_file_data(train_f)
        self.test_var, self.test_output = self.read_file_data(test_f)


    @staticmethod
    def read_file_data(file):
        """
            Given a file returns two dataframes, one with its
            variables and another one with outputs
        """
        csv_file = open(file, 'r')
        reader = list(csv.reader(csv_file))[20:]
        reader = np.array(reader)

        variables = np.array(
            list(list(map(lambda x: np.nan if x == 'na' else np.float(x), l))
                 for l in reader[1:, 1:]))
        output = np.fromiter(map(lambda x: -1 if x == "neg" else 1,
                                 reader[1:, 0]), dtype=np.int)
        variables = pd.DataFrame(variables)

        csv_file.close()

        return variables, output


    def preprocess(self, show_evolution=True, normalization=True):
        """
            Preprocessing of training and test data.
        """
        print("Preprocesamiento de datos para clasificación")

        #print(self.train_var)

        # Impute missing values
        # TODO: si la estrategia de imputación es la de la media, se puede simplemente poner los NaN a 0 tras normalizar
        self.__impute_missing_values()

        # Normalization
        self.__normalize()

        # PCA
        #self.__pca(10)  # TODO: discutir número de componentes


    def __impute_missing_values(self):
        """
            Impute missing values
        """
        # TODO: discutir estrategia de imputación
        imp = SimpleImputer()
        imp.fit(self.train_var)
        self.train_var = pd.DataFrame(imp.transform(self.train_var))
        self.test_var = pd.DataFrame(imp.transform(self.test_var))


    def __normalize(self):
        """
            Normalize data
        """
        # TODO: discutir tipo de normalización

        # Transform data so that it has mean 0 and s.d. 1
        sc = StandardScaler()
        sc.fit(self.train_var)
        self.train_var = pd.DataFrame(sc.transform(self.train_var))
        self.test_var = pd.DataFrame(sc.transform(self.test_var))


    def __pca(self, n_components):
        #print(self.train_var.shape)
        pca = PCA(n_components=n_components)
        pca.fit(self.train_var)
        self.train_var = pd.DataFrame(pca.transform(self.train_var))
        self.test_var = pd.DataFrame(pca.transform(self.test_var))
        #print(self.train_var.shape)


    def remove_indexes(self, data, idx):
        """
            Given a dataframe it returns a new dataframe with 
            columns idx removed
        """

        idx = set(idx)
        new_matrix = []
        n = len(data.columns)
        for column in range(n):
            if column not in idx:
                new_matrix.append(data.iloc[:,column])

        new_matrix = pd.DataFrame(new_matrix).T
        return new_matrix


    def increase_var_pol(self, degree=1, interaction_only=True):
        """ Transform dimensionaliy of our dataset
        Suppose variables [a,b] 
            If we choose degree=2 and interaction_only=False it will change to: [1, a, b, a^2, ab, b^2]
            If we choose degree=2 and interaction_only=True it will change to: [1, a, b, ab]
        """
        poly = PolynomialFeatures(degree, interaction_only=interaction_only)
        print("Aumento de dimensionalidad: " + str(self.train_var.shape[1]) + " -> ", end=" ")
        self.train_var = pd.DataFrame(poly.fit_transform(self.train_var))
        self.test_var = pd.DataFrame(poly.fit_transform(self.test_var))
        print(self.train_var.shape[1])


    # Getter
    def get_sample_weight(self, train=True):
        if train:
            return sklearn.utils.class_weight.compute_sample_weight(DataSet.weights_dic, self.train_output)
        else:
            return sklearn.utils.class_weight.compute_sample_weight(DataSet.weights_dic, self.test_output)


    def get_sample_weight_train(self):
        return self.get_sample_weight()
        

    def get_sample_weight_test(self):
        return self.get_sample_weight(train=False)      


    # Plotting functions

    def plot(self, i, j):
        """
            Plot train variables (i,j)
        """
        plt.scatter(self.train_var.iloc[:,i], self.train_var.iloc[:,j])
        plt.show()


    def nan_histogram(self):
        xs = sum(np.array(list(map(lambda x: np.isnan(x),
                                   self.train_var.values)))) \
             / len(self.train_var)
        xs.sort()
        plt.hist(xs, 20, density=False)
        plt.xlabel("Porcentaje de valores desconocidos en una variable")
        plt.ylabel("Porcentaje de variables")
        plt.title("Histograma de la distribución de valores desconocidos por "
                  "variables")
        plt.show()


def get_dataset(small=False):
    data_folder = "./datos"
    if small:
        train_f, test_f = "reduced_training_set.csv", "reduced_test_set.csv"
    else:
        train_f, test_f = "aps_failure_training_set.csv", "aps_failure_test_set.csv" 
    ds = DataSet(f"{data_folder}/{train_f}", f"{data_folder}/{test_f}") 
    return ds
