import pandas as pd
import random

class DrugPricingDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples
    
    def generate_data_provider_dataset(self):
        data = []
        for _ in range(self.num_samples):
            feature1 = random.uniform(0.5, 2.5)  # 特征1：药品成本
            feature2 = random.uniform(1.0, 3.0)  # 特征2：研发成本
            feature3 = random.uniform(0.8, 1.2)  # 特征3：市场需求
            feature4 = random.uniform(0.5, 1.5)  # 特征4：竞争对手价格
            feature5 = random.uniform(0.8, 1.2)  # 特征5：药品效果
            feature6 = random.uniform(0.5, 2.0)  # 特征6：法规限制
            feature7 = random.uniform(0.8, 1.2)  # 特征7：营销费用
            price = (
                2 * feature1
                + 3 * feature2
                + 1.5 * feature3
                - 1.2 * feature4
                + 0.5 * feature5
                - 0.8 * feature6
                + 1.2 * feature7
            )  # 定价：线性组合
            
            data.append([
                feature1, feature2, feature3, feature4, feature5, feature6, feature7, price
            ])
        
        df = pd.DataFrame(
            data,
            columns=[
                'Feature1', 'Feature2', 'Feature3', 'Feature4',
                'Feature5', 'Feature6', 'Feature7', 'Price'
            ]
        )
        df.to_csv('C:/Users/24090/Desktop/FATE-FairPricing/data_provider_dataset.csv', index=False)
    
    def generate_data_buyer_dataset(self):
        data = []
        for _ in range(self.num_samples):
            feature1 = random.uniform(0.5, 2.5)  # 特征1：药品成本
            feature2 = random.uniform(1.0, 3.0)  # 特征2：研发成本
            feature3 = random.uniform(0.8, 1.2)  # 特征3：市场需求
            feature4 = random.uniform(0.5, 1.5)  # 特征4：竞争对手价格
            feature5 = random.uniform(0.8, 1.2)  # 特征5：药品效果
            feature6 = random.uniform(0.5, 2.0)  # 特征6：法规限制
            feature7 = random.uniform(0.8, 1.2)  # 特征7：营销费用
            
            data.append([
                feature1, feature2, feature3, feature4, feature5, feature6, feature7
            ])
        
        df = pd.DataFrame(
            data,
            columns=[
                'Feature1', 'Feature2', 'Feature3', 'Feature4',
                'Feature5', 'Feature6', 'Feature7'
            ]
        )
        df.to_csv('C:/Users/24090/Desktop/FATE-FairPricing/data_buyer_dataset.csv', index=False)

# 示例使用方式
num_samples = 10000  # 数据样本数量

data_generator = DrugPricingDataGenerator(num_samples)

# 生成数据提供方的虚拟数据集
data_generator.generate_data_provider_dataset()

# 生成数据购买方的虚拟数据集
data_generator.generate_data_buyer_dataset()