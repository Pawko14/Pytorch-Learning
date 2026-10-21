import torch
import matplotlib.pyplot as plot
from torch import nn as nn 
from plotPredictions import plot_preditions as pp

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device) # using GPU

#1. Data

# ax^2 + bx + c formula
a = 0.5
b = 0.7
c = 0.3

start = -5
end = 5
step = 0.02

# Creating a tensor with a 2 column 
# 1st column needs a x^2 
# 2nd column needs a x 

X1 = torch.arange(start, end, step).unsqueeze(dim = 1)
X_Squared = X1 ** 2

X = torch.cat((X_Squared, X1), dim = 1)

# print(X)
y = a * X[:,0:1] + b*X[:,1:2] + c

# print(f"y: {y}")

test_size = (int) (0.8 * len(X))

X_train = X[:test_size]
X_test = X[test_size:]

y_train = y[:test_size]
y_test = y[test_size:]

class Regresion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype= torch.float))
        self.weight2 = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype= torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype = torch.float))
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.weight1*X[:, 0:1] + self.weight2*X[:, 1:2] + self.bias

model = Regresion()

print(len(X_train) , len(y_train) , len(X_test) , len(y_test))


print(model.state_dict())

with torch.inference_mode():
    y_predictions = model(X_test)


pp(X_train[:,1:2], y_train, X_test[:,1:2], y_test, y_predictions)

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)

epochs = 1000

for epoch in range(epochs):
    model.train()
    y_new_pred = model(X_train)
    loss = loss_fn(y_new_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.inference_mode():
        y_last = model(X_test)
        last_loss = loss_fn(y_last,y_test)

pp(X_train[:,1:2], y_train, X_test[:,1:2], y_test, y_last)

print(model.state_dict())