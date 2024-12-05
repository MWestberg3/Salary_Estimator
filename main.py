from DataGenerator import DataGenerator
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Instantiate DataGenerators
current_salary_list = []
salary_history_mean_list = []
salary_std_dev_list = []
tenure_list = []
inflation_rate_mean_list = []
inflation_rate_std_dev_list = []
current_bonus_list = [] # May not matter

for i in range(300):
    tenure = random.randint(1, 20)
    current_bonus = random.uniform(1000, 10000)
    average_growth_rate = random.uniform(0.01, 0.04)
    variability = 0.02
    current_salary = random.uniform(50000, 150000)

    generated_data = DataGenerator(tenure, current_salary, current_bonus, average_growth_rate, variability)

    np_salary_history = np.array(generated_data.salary_history)

    salary_mean = round(np.mean(np_salary_history), 2).item()
    salary_std_dev = round(np.std(np_salary_history), 2).item()

    inflation_rate_mean = round(np.mean(generated_data.inflation_rates), 2).item()
    inflation_rate_std_dev = round(np.std(generated_data.inflation_rates), 2).item()

    current_salary_list.append(round(current_salary, 2))
    salary_history_mean_list.append(salary_mean)
    salary_std_dev_list.append(salary_std_dev)
    tenure_list.append(tenure)
    inflation_rate_mean_list.append(inflation_rate_mean)
    inflation_rate_std_dev_list.append(inflation_rate_std_dev)

# Format data to be a numpy array for the RandomForestClassifier
X = np.array([salary_history_mean_list, salary_std_dev_list, tenure_list, inflation_rate_mean_list, inflation_rate_std_dev_list]).T
y = np.array(current_salary_list)
print(len(X))
print(len(y))


### GEEKS FOR GEEKS EXAMPLE ON IRIS DATASET ###
# iris = load_iris()

# X, y = load_iris(return_X_y=True)

# print(type(X))
# print(X)
# print(len(y))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# try:
#     sepalwidth = iris.data[:, 1]
# except IndexError:
#     sepalwidth = None

# data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': sepalwidth, 'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],'target': iris.target})

# print(data.head())

# clf = RandomForestClassifier(n_estimators=100)

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# print()
# print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
# print(feature_imp)