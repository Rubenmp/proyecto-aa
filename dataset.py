#############
#  DATASET  #
#############

import csv
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.manifold import TSNE
import time

class DataSet:
    POS_CLASS_WEIGHT = 500
    NEG_CLASS_WEIGHT = 10
    POSITIVE_CLASS = 1
    NEGATIVE_CLASS = -1
    WEIGHTS_DIC = {POSITIVE_CLASS: POS_CLASS_WEIGHT,
                   NEGATIVE_CLASS: NEG_CLASS_WEIGHT}
    
    def __init__(self, train_f, test_f):
        """
            Initialize dataset from file(s)
        """
        self.train_var, self.train_output = None, None
        self.test_var, self.test_output = None, None
        self.var_names = None

        start_time = time.time()
        self.read_train_test(train_f, test_f)
        print("--- %s seconds ---" % (time.time() - start_time))

    def read_train_test(self, train_f, test_f):
        """
            Read training and test data from files
        """
        self.train_var, self.train_output = self.read_file_data(train_f)
        self.test_var, self.test_output = self.read_file_data(test_f)

    def read_file_data(self, file):
        """
            Given a file returns two dataframes, one with its
            variables and another one with outputs
        """
        csv_file = open(file, 'r')
        reader = list(csv.reader(csv_file))[20:]
        reader = np.array(reader)

        self.var_names = [l for l in reader[0, 1:]]

        variables = np.array(
            list(list(map(lambda x: np.nan if x == 'na' else np.float(x), l))
                 for l in reader[1:, 1:]))
        output = np.fromiter(map(lambda x: -1 if x == "neg" else 1,
                                 reader[1:, 0]), dtype=np.int)

        csv_file.close()

        return variables, output

    def preprocess(self, show_evolution=True, normalization=True):
        """
            Preprocessing of training and test data.
        """
        print("Preprocesamiento de datos para clasificación")

        # Impute missing values

        # TODO: si la estrategia de imputación es la de la media, se puede simplemente poner los NaN a 0 tras normalizar
        start_time = time.time()
        self.__impute_missing_values()
        print("--- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        self.__remove_outliers()
        print("--- %s seconds ---" % (time.time() - start_time))

        if normalization:
            self.__normalize()
        


    def __impute_missing_values(self):
        """
            Impute missing values
        """
        # TODO: discutir estrategia de imputación
        imp = SimpleImputer()
        imp.fit(self.train_var)
        self.train_var = imp.transform(self.train_var)
        self.test_var = imp.transform(self.test_var)

    def __remove_outliers(self, show_evolution=False):
        """
            Remove outliers from data with class obj_class
            If outliers are removed from data not taking into
            account obj_class unbalanced classes could be considered
            as outliers 
        """

        if show_evolution:
            print("Before. Train shape " + str(self.train_var.shape) + ", outputs " + str(len(self.train_output)))
        
        # Remove outliers with negative class
        neg_class_data = DataSet.__remove_outliers_by_class(self.train_var, self.train_output, DataSet.NEGATIVE_CLASS)
        neg_classes = [DataSet.NEGATIVE_CLASS for _ in neg_class_data]

        pos_class_data = DataSet.__remove_outliers_by_class(self.train_var, self.train_output, DataSet.POSITIVE_CLASS)
        pos_classes = [DataSet.POSITIVE_CLASS for _ in pos_class_data]

        data = np.concatenate((neg_class_data, pos_class_data))
        classes = np.concatenate((neg_classes, pos_classes))

        # Shuffle data
        random_indexes = [i for i in range(len(classes))]
        np.random.shuffle(random_indexes)

        self.train_var = data[random_indexes]
        self.train_output = classes[random_indexes]

        if show_evolution:
            print("After. Train shape " + str(self.train_var.shape) + ", outputs " + str(len(self.train_output)))

    @staticmethod
    def __remove_outliers_by_class(data, classes, obj_class):
        """
            Remove outliers from data with class obj_class
            If outliers are removed from data not taking into
            account obj_class unbalanced classes could be considered
            as outliers 
        """
        # Parameters of Isolation Forest are default parameters of 
        # new version, if we do not set these parameters it would
        # be a deprecated version of IsolationForest
        num_var = len(data[0])
        outliers_IF = IsolationForest(behaviour="new", contamination="auto", n_estimators=num_var)
        data_with_obj_class = data[classes == obj_class]
        outliers_IF.fit(data_with_obj_class)

        # Inliers are labeled 1, while outliers are labeled -1.
        y_pred   = outliers_IF.predict(data_with_obj_class)
        new_data = data_with_obj_class[y_pred != -1]

        return new_data

    def __normalize(self):
        """
            Normalize data
        """
        # TODO: discutir tipo de normalización

        # Transform data so that it has mean 0 and s.d. 1
        sc = StandardScaler()
        sc.fit(self.train_var)
        self.train_var = sc.transform(self.train_var)
        self.test_var  = sc.transform(self.test_var)

    # Getter
    def get_sample_weight(self, train=True):
        if train:
            return compute_sample_weight(DataSet.WEIGHTS_DIC, self.train_output)
        else:
            return compute_sample_weight(DataSet.WEIGHTS_DIC, self.test_output)

    def get_sample_weight_train(self):
        return self.get_sample_weight()

    def get_sample_weight_test(self):
        return self.get_sample_weight(train=False)

    # Plotting functions
    def scatter(self, show=True):
        colors = {DataSet.POSITIVE_CLASS: 'red', DataSet.NEGATIVE_CLASS: 'blue'}
        label_colors = [colors[label] for label in self.train_output]

        # Projection of training data into two dimensions
        X_embedded = TSNE(n_components=2, method='exact', random_state=0).fit_transform(self.train_var)
        var1 = [x[0] for x in X_embedded]
        var2 = [x[1] for x in X_embedded]

        # Plot
        plt.title(f'Visualización de APS Failure at Scania Trucks Data Set')
        plt.scatter(var1, var2, s=3, c=label_colors)

        if show:
            plt.show()


    def plot_boxplot(self, data, index, show_outliers=False, file=None):
        sns.boxplot(x=data[index], showfliers=show_outliers)
        plt.title(f'Boxplot de la variable {self.var_names[index]}')
        if file is not None:
            plt.savefig(file)
        plt.show()

    def nan_histogram(self):
        xs = sum(np.array(list(map(lambda x: np.isnan(x),
                                   self.train_var)))) \
             / len(self.train_var)
        xs.sort()
        plt.hist(xs, 20, density=False)
        plt.xlabel("Porcentaje de valores desconocidos en una variable")
        plt.ylabel("Porcentaje de variables")
        plt.title("Histograma de la distribución de valores desconocidos por "
                  "variables")
        plt.show()


def get_sample_weight(y_true):
    w_dic = DataSet.WEIGHTS_DIC
    sample_weight=compute_sample_weight(w_dic, y_true)
    return sample_weight


def get_aps_dataset(small=False):
    data_folder = "./datos"
    if small:
        train_f, test_f = "reduced_training_set.csv", "reduced_test_set.csv"
    else:
        train_f, test_f = "aps_failure_training_set.csv", "aps_failure_test_set.csv" 
    ds = DataSet(f"{data_folder}/{train_f}", f"{data_folder}/{test_f}")

    return ds
