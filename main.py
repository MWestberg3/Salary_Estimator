import DataGenerator
from sklearn.ensemble import RandomForestClassifier

# Create a DataGenerator object
data_generator = DataGenerator.DataGenerator(tenure=10, current_salary=100000, current_bonus=10000, average_growth_rate=0.05, variability=0.02)