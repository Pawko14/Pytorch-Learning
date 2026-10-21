import torch
from torch import nn as nn 
import matplotlib.pyplot as plot
import plotPredictions as PP

# We're making formula y = a * x1 + b * x2 + c

X_first = torch.randint(30,120,(100,)).unsqueeze(dim = 1)



X_second = torch.randint(1,5,(100,)).unsqueeze(dim = 1)

X = torch.cat((X_first,X_second),1).float() # first column - X1(random number from 30 to 120) second column - X2(random number from 1 to 5)

X[:, 0:1] = X[: ,0:1] / 120
X[:, 1:2] = X[: ,1:2] / 5

a = 0.7
b = 0.5
c = 0.3

y = a * X[: , 0:1] + b * X[: , 1:2] + c

train_data = (int)(0.8 * len(X))

X_train = X[:train_data]
X_test = X[train_data:]

y_train = y[:train_data]
y_test = y[train_data:]


class MultipleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(1, 
                                                dtype=torch.float,
                                                requires_grad=True)
                                                )
        self.weight2 = nn.Parameter(torch.randn(1, 
                                                dtype=torch.float,
                                                requires_grad=True)
                                                )
        self.bias = nn.Parameter(torch.randn(1, 
                                                dtype=torch.float,
                                                requires_grad=True)
                                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight1 * x[: , 0:1] + self.weight2 * x[: , 1:2] + self.bias

model = MultipleLinearRegression()

print(model.state_dict())

with torch.inference_mode():
    y_start_pred = model(X_test)

loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

epochs = 50000

for epoch in range(epochs):
    model.train() 

    y_1 = model(X_train)
    loss = loss_fn(y_1, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_end_pred = model(X_test)
        test_loss = loss_fn(y_end_pred,y_test)

        # if epoch % 10 == 0:
            # print(f"Parametry: epoka {epoch}. Wartosci: {model.state_dict()}")

print(a , b , c)
print(model.state_dict())


