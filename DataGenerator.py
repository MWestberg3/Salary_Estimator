import numpy as np

class DataGenerator:
    def __init__(self, tenure, current_salary, current_bonus, average_growth_rate, variability):
        self.tenure = tenure
        self.salary_history = self.generate_salary_data(current_salary, tenure, average_growth_rate, variability)
        self.bonus_history = self.generate_bonus_data(tenure, current_bonus, average_growth_rate, variability)
        self.inflation_rates = self.generate_inflation_rates(tenure)

    # Generate previous salary data with random growth rate in reverse
    def generate_salary_data(current_salary, tenure, average_growth_rate, variability):
        salaries = [current_salary]
        for year in range(1, tenure):
            random_growth = np.random.uniform(average_growth_rate - variability, average_growth_rate + variability)
            new_salary = salaries[-1] / (1 + random_growth)
            salaries.append(new_salary)
        return np.array(salaries)

    # Generate Bonus Data
    def generate_bonus_data(tenure, current_bonus, growth_rate, variability):
        bonuses = [current_bonus]
        for year in range(1, tenure):
            random_growth = np.random.uniform(growth_rate - variability, growth_rate + variability)
            new_bonus = bonuses[-1] * (1 + random_growth)
            bonuses.append(new_bonus)
        return np.array(bonuses)

    # Generate yearly inflation rates
    def generate_inflation_rates(tenure, min_rate=0.012, max_rate=0.083):
        return np.random.uniform(min_rate, max_rate, tenure)
    
    def generate_performance_review():
        pass


