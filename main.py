from DataGenerator import DataGenerator
from sklearn.ensemble import RandomForestRegressor
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

# Instantiate DataGenerators
current_salary_list = []
salary_history_mean_list = []
salary_history_std_dev_list = []
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
    salary_history_std_dev_list.append(salary_std_dev)
    tenure_list.append(tenure)
    inflation_rate_mean_list.append(inflation_rate_mean)
    inflation_rate_std_dev_list.append(inflation_rate_std_dev)

# Format data to be a numpy array for the RandomForestClassifier
X = np.array([salary_history_mean_list, salary_history_std_dev_list, tenure_list, inflation_rate_mean_list, inflation_rate_std_dev_list]).T
y = np.array(current_salary_list)

data = pd.DataFrame({'salary_history_mean': salary_history_mean_list, 'salary_history_std_dev': salary_history_std_dev_list, 'tenure': tenure_list, 'inflation_rate_mean': inflation_rate_mean_list, 'inflation_rate_std_dev': inflation_rate_std_dev_list, 'current_salary': current_salary_list})

print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

param_grid = {
    'n_estimators': [100, 200, 300], # Common values for n_estimators
    'max_depth': [None, 10, 20, 30], # Common values for max_depth
    'min_samples_split': [2, 5, 10], # Common values for min_samples_split
    'min_samples_leaf': [1, 2, 4], # Common values for min_samples_leaf
    'max_features': ['auto', 'sqrt', 'log2'], # Common values for max_features
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# clf = RandomForestRegressor(n_estimators=100)

# clf.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
# y_pred = clf.predict(X_test)

print()
mae = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

best_model = grid_search.best_estimator_
feature_imp = pd.Series(best_model.feature_importances_, index=['salary_history_mean', 'salary_history_std_dev', 'tenure', 'inflation_rate_mean', 'inflation_rate_std_dev']).sort_values(ascending=False)
print(feature_imp)

# feature_imp = pd.Series(clf.feature_importances_, index=['salary_history_mean', 'salary_history_std_dev', 'tenure', 'inflation_rate_mean', 'inflation_rate_std_dev']).sort_values(ascending=False)
# print(feature_imp)