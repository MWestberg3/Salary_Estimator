import numpy as np

class DataGenerator:
    def __init__(self, tenure: int, current_salary: float, current_bonus: float, average_growth_rate: float, variability: float):
        self.tenure = tenure
        current_and_historical_salary = self.generate_salary_data(current_salary, tenure, average_growth_rate, variability)
        self.salary_history = current_and_historical_salary[1:]
        self.bonus_history = self.generate_bonus_data(tenure, current_bonus, average_growth_rate, variability)
        self.inflation_rates = self.generate_inflation_rates(tenure)

    # Generate previous salary data with random growth rate in reverse
    def generate_salary_data(self, current_salary: float, tenure: int, average_growth_rate: float, variability: float) -> list[float]:
        salaries = [current_salary]
        for year in range(tenure):
            random_growth = np.random.uniform(average_growth_rate - variability, average_growth_rate + variability)
            new_salary = round(salaries[-1] / (1 + random_growth), 2)
            salaries.append(new_salary)
        return salaries

    # Generate Bonus Data
    def generate_bonus_data(self, tenure: int, current_bonus: float, growth_rate: float, variability: float) -> list[float]:
        bonuses = [current_bonus]
        for year in range(tenure):
            random_growth = np.random.uniform(growth_rate - variability, growth_rate + variability)
            new_bonus = round(bonuses[-1] * (1 + random_growth), 2)
            bonuses.append(new_bonus)
        return bonuses

    # Generate yearly inflation rates
    def generate_inflation_rates(self, tenure: int, min_rate=0.012, max_rate=0.083) -> list[float]:
        return np.random.uniform(min_rate, max_rate, tenure)
    
    def generate_performance_review():
        pass


