import torch
import numpy as np
import matplotlib.pyplot as plt



x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    '''逻辑回归模型'''
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = x.float()
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data.float())
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)


x = np.linspace(0, 10, 200)
x_t = torch.tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()


# 测试git
# tetst
