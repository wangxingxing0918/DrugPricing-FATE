import pandas as pd
from federatedml import LogisticRegression
from federatedml.data import FederatedDataSet
from federatedml.util import consts
from sklearn.preprocessing import MinMaxScaler
from federatedml.nn.hetero_dnn.hetero_dnn_model import HeteroDNN
from model import ComplexModel
import argparse


class FederatedLearning:
    def __init__(self, guest_data_path, host_data_path, learning_rate=0.01, num_epochs=10, batch_size=32, hidden_units=[64, 32], dropout_rate=0.2):
        self.guest_data_path = guest_data_path
        self.host_data_path = host_data_path
        self.guest_data = None
        self.host_data = None
        self.guest_train_data = None
        self.host_train_data = None
        self.guest_val_data = None
        self.host_val_data = None
        self.model = None
        self.model_param = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate
        }
        
    def load_data(self):
        self.guest_data = pd.read_csv(self.guest_data_path)
        self.host_data = pd.read_csv(self.host_data_path)

    def preprocess_data(self):
        # 数据预处理逻辑
        # 划分训练集和验证集
        self.guest_train_data = self.guest_data.sample(frac=0.8, random_state=42)
        self.guest_val_data = self.guest_data.drop(self.guest_train_data.index)
        self.host_train_data = self.host_data.sample(frac=0.8, random_state=42)
        self.host_val_data = self.host_data.drop(self.host_train_data.index)

        # 特征工程
        # 对guest_train_data进行特征工程处理
        self.guest_train_data = self.feature_engineering(self.guest_train_data)

        # 对guest_val_data进行特征工程处理
        self.guest_val_data = self.feature_engineering(self.guest_val_data)

        # 对host_train_data进行特征工程处理
        self.host_train_data = self.feature_engineering(self.host_train_data)

        # 对host_val_data进行特征工程处理
        self.host_val_data = self.feature_engineering(self.host_val_data)

        # 数据归一化
        self.guest_train_data = self.normalize_data(self.guest_train_data)
        self.guest_val_data = self.normalize_data(self.guest_val_data)
        self.host_train_data = self.normalize_data(self.host_train_data)
        self.host_val_data = self.normalize_data(self.host_val_data)

    def feature_engineering(self, data):
        # 特征工程逻辑

        # 添加新的特征
        data['new_feature'] = data['feature1'] + data['feature2']

        # 删除不需要的特征
        data = data.drop(['feature1', 'feature2'], axis=1)

        return data


    def normalize_data(self, data):
        # 创建MinMaxScaler对象
        scaler = MinMaxScaler()

        # 将数据进行归一化处理
        data_normalized = scaler.fit_transform(data)

        return data_normalized

    def train_model(self):
        # 创建FederatedDataSet
        guest_train_data = FederatedDataSet(self.guest_train_data, self.guest_val_data)
        host_train_data = FederatedDataSet(self.host_train_data, self.host_val_data)

        # 创建ComplexModel模型
        self.model = ComplexModel()

        # 设置训练参数
        self.model.fit(guest_train_data, host_train_data, self.model_param)

    def predict(self, data):
        # 使用训练好的模型进行预测
        return self.model.predict(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--guest_data_path', type=str, help='Path to guest data')
    parser.add_argument('--host_data_path', type=str, help='Path to host data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_units', type=list, default=[64, 32], help='Hidden units')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()

    fl = FederatedLearning(args.guest_data_path, args.host_data_path, args.learning_rate, args.num_epochs, args.batch_size, args.hidden_units, args.dropout_rate)
    fl.load_data()
    fl.preprocess_data()
    fl.train_model()
    
    # 使用模型进行预测
    data_to_predict = pd.read_csv('data_to_predict.csv')
    prediction = fl.predict(data_to_predict)
    # 使用示例
    guest_data_path = 'guest_data.csv'
    host_data_path = 'host_data.csv'
    
    fl = FederatedLearning(guest_data_path, host_data_path, learning_rate=0.01, num_epochs=10, batch_size=32, hidden_units=[64, 32], dropout_rate=0.2)
    fl.load_data()
    fl.preprocess_data()
    fl.train_model()
    
    # 使用模型进行预测
    data_to_predict = pd.read_csv('data_to_predict.csv')
    prediction = fl.predict(data_to_predict)