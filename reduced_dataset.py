import numpy as np

train_file = open('./datos/aps_failure_training_set.csv', 'r')
test_file = open('./datos/aps_failure_test_set.csv')

reduced_train_file = open('./datos/reduced_training_set.csv', 'w')
reduced_test_file = open('./datos/reduced_test_set.csv', 'w')

train = np.array(train_file.readlines())
test = np.array(test_file.readlines())

reduced_train_file.writelines(train[:21])
reduced_train_file.writelines(np.random.choice(train[21:], 6000, replace=False))
reduced_test_file.writelines(test[:21])
reduced_test_file.writelines(np.random.choice(test[21:], 1000, replace=False))

train_file.close()
test_file.close()
reduced_train_file.close()
reduced_test_file.close()
