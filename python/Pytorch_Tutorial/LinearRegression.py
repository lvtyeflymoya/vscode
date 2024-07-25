import torch


# print(torch.cuda.is_available())
# print(torch.__version__)

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])



class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调用父类（LinearModel的父类）的初始化方法
        super(LinearModel, self).__init__()

        # 创建一个线性层，输入特征数为1，输出特征数也为1，包含了权重和偏置项
        self.linear = torch.nn.Linear(1, 1)  # weights and bias
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

# criterion = torch.nn.MSELoss(size_average=False,reduce='sum')
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    # 更新权重

# output weights and bias
print('w = ',round(model.linear.weight.item(),6))
print('b = ',round(model.linear.bias.item(),6))

# print('w = ',model.linear.weight.item())
# print('b = ',model.linear.bias.item())

# test model
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)


# git test

