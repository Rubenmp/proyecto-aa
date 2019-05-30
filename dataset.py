#############
#  DATASET  #
#############

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

class DataSet:
    def __init__(self, train_f, test_f):
        """
            Initialize dataset from file(s)
        """
        self.train_var, self.train_output = None, None
        self.test_var, self.test_output = None, None

        self.read_train_test(train_f, test_f)

    def read_train_test(self, train_f, test_f):
        """
            Read training and test data directly from two files
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

        names = reader[0, 1:]
        variables = np.array(
            list(list(map(lambda x: np.nan if x == 'na' else np.float(x), l))
                 for l in reader[1:, 1:]))
        output = np.fromiter(map(lambda x: -1 if x == "neg" else 1,
                                 reader[1:, 0]), dtype=np.int)
        variables = pd.DataFrame(variables)
        variables.columns = names
        output = pd.DataFrame(output)

        return variables, output

    def preprocess(self, show_evolution=True, normalization=True):
        """
            Preprocessing of training and test data.
        """

        # Impute missing values
        self.__impute_missing_values()

        # if show_evolution:
        #     print("\tEvolución del número de variables en el preprocesamiento")
        #     if normalization:
        #         print("\tNormalización: " + str(self.train_var.shape[1]), end=" ")
        # if normalization:
        #     self.__normalize()
        #
        #
        # cor_threshold = 0.8
        # if show_evolution:
        #     if normalization:
        #         print(" -> " + str(self.train_var.shape[1]))
        #     print("\tEliminar variables con correlación superior a " + str(cor_threshold) + ": " + str(self.train_var.shape[1]), end=" ")
        #
        # self.train_var, self.test_var = self.__red_correlations(cor_threshold)
        #
        # var_threshold = 0.1
        # if show_evolution:
        #     print("-> " + str(self.train_var.shape[1]))
        #     print("\tEliminar variables con varianza inferior a " + str(var_threshold) + ": " + str(self.train_var.shape[1]), end=" ")
        #
        # self.train_var, self.test_var = self.__red_var(var_threshold)
        # if show_evolution:
        #     print(" -> " + str(self.train_var.shape[1]))

    def __impute_missing_values(self):
        # TODO: discutir estrategia de imputación
        imp = SimpleImputer()
        self.train_var = pd.DataFrame(imp.fit_transform(self.train_var))
        self.test_var = pd.DataFrame(imp.fit_transform(self.test_var))

    def __normalize(self):
        """
            Normalize data between [0,1]
        """

        # Normaliza train data
        rows, _ = self.train_var.shape
        new_train = []

        n = len(self.train_var.columns)
        remove_idx = []
        for column in range(n):
            max = float(self.train_var.iloc[:,column].max())
            min = float(self.train_var.iloc[:,column].min())

            new_column = []
            for row in range(rows):
                if min != max:
                    value = (self.train_var.iloc[row, column])/(max-min)
                else:
                    value = 0
                new_column.append(value)
            new_train.append(new_column)

        self.train_var = pd.DataFrame(new_train).T


        # Normalize test data
        rows, n = self.test_var.shape
        new_test = []

        for column in range(n):
            max = float(self.test_var.iloc[:,column].max())
            min = float(self.test_var.iloc[:,column].min())

            if min != max:
                new_column = []
                for row in range(rows):
                    value = (self.test_var.iloc[row, column])/(max-min)
                    new_column.append(value)
                new_test.append(new_column)
            else:
                # We can not use data from test to deduce which 
                # indexes should be removed, so we keep a constant
                # column
                new_column = []
                for row in range(rows):
                    value = 0
                    new_column.append(value)
                new_test.append(new_column)
        self.test_var = pd.DataFrame(new_test).T



    def __red_var(self, threshold):
        """
            Remove variables with variance < threshold
        """        
        idx = []

        n = len(self.train_var.columns)
        for column in range(n):
            var = self.train_var.iloc[:,column].var()
            if var < threshold:
                idx.append(column)

        yield self.remove_indexes(self.train_var, idx)
        yield self.remove_indexes(self.test_var, idx)


    def __red_correlations(self, threshold):
        """
            Remove variables which correlates > threshold with respect to others
        """

        # Compute indexes with high correlation
        corr_matrix = self.train_var.corr()
        remove_idx  = []
        n = corr_matrix.shape[0]

        n = len(corr_matrix.columns)
        for i in range(n):
            for j in range(n):
                if i != j and corr_matrix.iloc[i,j] > threshold:
                    if i not in remove_idx:
                        remove_idx.append(i)

        yield self.remove_indexes(self.train_var, remove_idx)
        yield self.remove_indexes(self.test_var, remove_idx)


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
        self.test_var  = pd.DataFrame(poly.fit_transform(self.test_var))
        print(self.train_var.shape[1])


    # Getters

    def get_item(self, i, train_ds=True):
        """
            Get item with index i in training/test data 
            depending on variable train_ds
        """
        if train_ds:
            return self.train_var.iloc[i,:], self.train_output.iloc[i,:]
        else:
            return self.test_var.iloc[i,:], self.test_output.iloc[i,:]


    def get_num_items(self, train_ds=True):
        """
            Get number of items in training/test data 
            depending on variable train_ds
        """
        if train_ds:
            return self.train_var.shape[0]
        else:
            return self.test_var.shape[0]


    def get_output(self, i, train_ds=True):
        """
            Get output of specific index i in training/test data 
            depending on variable train_ds
        """
        if train_ds:
            return self.train_output.iloc[i]
        else:
            return self.test_output.iloc[i]

    def get_train_data(self):
        """
            Get variables and output of training data
        """        
        n = self.get_num_items()
        X, y = [], []
        for i in range(n):
            new_X, new_y = self.get_item(i)
            X.append(list(new_X))
            y.append(list(new_y)[0])

        return X, y

    def get_test_data(self):
        """
            Get variables and output of test data
        """
        n = len(self.test_var)
        X, y = [], []
        for i in range(n):
            new_X, new_y = self.get_item(i, train_ds=False)
            X.append(list(new_X))
            y.append(list(new_y)[0])

        return X, y


    # Plotting functions

    def plot(self, i, j):
        """
            Plot train variables (i,j)
        """
        plt.scatter(self.train_var.iloc[:,i], self.train_var.iloc[:,j])
        plt.show()


    def plot_all(self):
        """
            Show relations between variables in training data.
            If variables are continuous the diagonal will show 
            an estimation of the probability density function,
            otherwise it will show histograms
        """

        df = self.train_var
        # Se muestran histogramas en la diagonal en
        # variables aleatorias discretas
        pd.plotting.scatter_matrix(df, diagonal='hist')
        plt.show()
