import DataGenerator
from sklearn.ensemble import RandomForestClassifier

tenure = 10
current_salary = 100000
current_bonus = 10000
average_growth_rate = 0.05
variability = 0.02

# Create a DataGenerator object
data_generator = DataGenerator.DataGenerator(tenure, current_salary, current_bonus, average_growth_rate, variability)

# Get the salary history
salary_history = data_generator.salary_history
print("Current Salary: ", current_salary)
print("Salary History: ", salary_history)

# Get the bonus history
bonus_history = data_generator.bonus_history
print("Bonus History: ", bonus_history)

# Get the inflation rates
inflation_rates = data_generator.inflation_rates
print("Inflation Rates: ", inflation_rates)

# Create a RandomForestClassifier object
rf_classifier = RandomForestClassifier()
