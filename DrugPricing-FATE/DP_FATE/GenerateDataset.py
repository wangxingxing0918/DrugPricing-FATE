from faker import Faker
import random
import pandas as pd

class CreditDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.fake = Faker()
    
    def generate_fake_data(self):
        data = []
        
        for _ in range(self.num_samples):
            age = random.randint(18, 65)
            income = random.randint(20000, 200000)
            debt = random.randint(5000, 100000)
            credit_score = random.randint(300, 850)
            employment_years = random.randint(0, 30)
            education_level = random.choice(['High School', 'Bachelor', 'Master', 'PhD'])
            default = random.choices([0, 1], weights=[0.8, 0.2])[0]
            
            row = [
                self.fake.unique.random_number(digits=6),
                age,
                income,
                debt,
                credit_score,
                employment_years,
                education_level,
                default
            ]
            data.append(row)
        
        return data
    
    def generate_dataset(self, filename):
        data = self.generate_fake_data()
        df = pd.DataFrame(
            data,
            columns=[
                'ID',
                'Age',
                'Income',
                'Debt',
                'CreditScore',
                'EmploymentYears',
                'EducationLevel',
                'Default'
            ]
        )
        df.to_csv(filename, index=False)
        print(f"已生成包含 {self.num_samples} 个样本的 {filename} 文件")

if __name__ == '__main__':
    # 使用 CreditDataGenerator 类生成包含 10000 个样本的数据集
    generator = CreditDataGenerator(10000)
    generator.generate_dataset('C:/Users/24090/Desktop/FATE-FairPricing/credit_data.csv')