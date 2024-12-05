import DataGenerator
from sklearn.ensemble import RandomForestClassifier
import random

# Instantiate DataGenerators
data_generator_list = []

for i in range(100):
    tenure = random.randint(1, 20)
    current_bonus = random.uniform(1000, 10000)
    average_growth_rate = random.uniform(0.01, 0.04)
    variability = 0.02
    current_salary = random.uniform(50000, 150000)

    data_generator_list.append(DataGenerator.DataGenerator(tenure, current_salary, current_bonus, average_growth_rate, variability))


# Create a RandomForestClassifier object
rf_classifier = RandomForestClassifier()
