import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    

dataset = DiabetesDataset('D:\\vscode\python\Pytorch_Tutorial\diabetes\diabetes.csv.gz')
train_loader = DataLoader(dataset, 
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()

if __name__ == '__main__':
    

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

data_test = DiabetesDataset('D:\\vscode\python\Pytorch_Tutorial\diabetes\diabetes_sample.csv')
# print(data_test[0])


# x_test = data_test[0][0]

# y_test = model(x_test)
# print('y_pred: ',y_test)
# ... 保持之前的代码不变 ...

if __name__ == '__main__':
    # ... 训练循环 ...

    # 加载测试数据集
    data_test = DiabetesDataset('D:\\vscode\python\Pytorch_Tutorial\diabetes\diabetes_sample.csv')

    # 假设我们只想要第一个测试样本
    x_test, y_test_real = data_test[0]  # 获取特征和真实标签
    y_test_real = y_test_real.item()  # 将标签转换为Python数字（如果它是一个单独的元素）

    # 不需要梯度计算，使用torch.no_grad()上下文
    with torch.no_grad():
        y_test_pred = model(x_test.unsqueeze(0))  # 增加一个维度以匹配模型的输入期望（batch_size, features）
        # 因为我们只传递了一个样本，所以需要unsqueeze(0)来添加一个batch维度

    # 打印预测结果（注意：这是sigmoid之后的输出，所以它在[0, 1]范围内）
    print('真实标签: ', y_test_real)
    print('预测值: ', y_test_pred.item())  # 假设我们只对第一个预测值感兴趣

# 注意：如果你的模型是为多类别分类设计的，并且你使用了softmax作为最后一层，
# 那么你应该使用torch.nn.CrossEntropyLoss而不是BCELoss，并且你的标签应该是类别索引（整数）。   
