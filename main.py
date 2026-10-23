import torch
from torch import nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd 
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_circles(n_samples, 
                    noise = 0.03,
                    random_state=42)

# print(f"pierwsze 5 X: \n {X[:5]}")
# print(f"pierwsze 5 y: \n {y[:5]}")

circles = pd.DataFrame({"X1" : X[:,0],
                        "X2" : X[:,1],
                        "label" : y})

print(circles.head(10))

print(circles.label.value_counts())

plt.scatter(x = X[:,0],
            y = X[:,1],
            c = y,
            cmap = plt.cm.RdYlBu);

# plt.show()

# print(X.shape , y.shape)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# print(X[:5] , " \n " , y[:5])

X_train, X_test , y_train, y_test = train_test_split(X ,y , test_size=0.2, random_state=42)

# print(f"train size: {len(X_train)}. \ntest size: {len(X_test)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2,out_features=5)
        self.layer2 = nn.Linear(in_features=5,out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))



# model1 = CircleModel()
model1 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features=5,out_features=1)
).to(device)
print(model1)

untrained_preds = model1(X_test.to(device))

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model1.parameters(), lr= 0.01)

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


y_start = model1(X_test.to(device))[:5]
print(y_start)

epochs = 1000

for epoch in range(epochs):

    model1.train()
    y1 = model1.forward(X_train)    
    loss = loss_fn(y1, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model1.eval()
    with torch.inference_mode():
        y_test_pred = model1(X_test)
        test_loss = loss_fn(y_test_pred,y_test)


    