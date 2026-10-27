import torch 
from torch import nn as nn
import sklearn
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from plot_decision_boundary import plot_decision_boundary as plt_dec 

n_samples = 1000
X, y = datasets.make_circles(n_samples = n_samples, noise = 0.03, random_state = 42)

# print(len(X), len(y))

# print(X[:5], " \n" , y[:5])

# circles = pd.DataFrame({"X0": X[:,0], "X1": X[:,1], "label": y})

# print(circles.head(10))

plt.scatter(x = X[:,0],
            y = X[:,1],
            c= y,
            cmap = plt.cm.RdYlBu)
# plt.show()

#Chcemy zrobic klasyfikacje 

device = "cuda" if torch.cuda.is_available() else "cpu"

# po mojemu to jest ale mozna tez zrobic przez sklearn

# train_percent = (int)(0.8 * n_samples)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# X_train, X_test = X[:train_percent] , X[train_percent:]
# y_train, y_test = y[:train_percent], y[train_percent:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(len(X_train) , " " , len(X_test), " " , X.dtype , " " , X.device)

# class Modelek(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(in_features = 2, out_features = 8)
#         self.layer2 = nn.Linear(in_features = 8, out_features = 1)

#     def forward(self, x):
#         return self.layer2(self.layer1(x)) # x -> layer1 -> layer2 -> output

# model1 = Modelek().to(device)

model1 = nn.Sequential(
        nn.Linear(in_features = 2, out_features = 128),
        nn.ReLU(),
        nn.Linear(in_features = 128, out_features = 256),
        nn.ReLU(),
        nn.Linear(in_features = 256, out_features = 128),
        nn.ReLU(),
        nn.Linear(in_features = 128, out_features = 1)
).to(device)

# print("\n", model1.state_dict())
with torch.inference_mode():
    untrained_preds = model1(X_test.to(device))

# print(len(untrained_preds) , " " , len(X_test) , " " , len(y_test))

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model1.parameters(),
                            lr = 0.001)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

# model1.eval()
# with torch.inference_mode():
#     y_logits = model1(X_test.to(device))[:5]
#     print(y_logits)

# y_preds_probs = torch.sigmoid(y_logits)
# y_preds = torch.round(y_preds_probs)

epochs = 100

X_train, X_test = X_train.to(device) , X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in range(epochs):
    model1.train()
    
    y1 = model1(X_train).squeeze() # logits 
    y_preds_probs = torch.sigmoid(y1) # 
    y_preds = torch.round(y_preds_probs)

    loss = loss_fn(y1, y_train)
    acc = accuracy_fn(y_train, y_preds)

    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()

    model1.eval()
    with torch.inference_mode():
        y_test_1 = model1(X_test).squeeze()
        y_test_preds_prob = torch.sigmoid(y_test_1)
        y_test_preds = torch.round(y_test_preds_prob)

        loss_test = loss_fn(y_test_1, y_test)
        acc1 = accuracy_fn(y_test, y_test_preds)
        if epoch % 10 == 0:
            print(loss_test , " " , acc1)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plt_dec(model1,X_train,y_train)

plt.subplot(1,2,2)
plt.title("Test")
plt_dec(model1, X_test,y_test)
plt.show()

n_samples = 10000

X1, y1 =datasets.make_moons(n_samples= n_samples, noise=0.1)

X1 = torch.from_numpy(X1).type(torch.float)
y1 = torch.from_numpy(y1).type(torch.float)
test_train = (int)(0.8 * n_samples)

X1_train = X1[:test_train]
X1_test = X1[test_train:]

y1_train = y1[:test_train]
y1_test = y1[test_train:]


print(len(X1_train), len((X1_test)))

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features= 2, out_features= 64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
    def forward(self,x):
        z = self.relu(self.layer1(x))
        z = self.relu(self.layer2(z))
        z = self.relu(self.layer3(z))
        z = self.layer4(z)
        return z

model2 = Model2().to(device)

loss2_fn = torch.nn.BCEWithLogitsLoss()
optimizer2 = torch.optim.Adam(params=model2.parameters(), lr = 0.01)

X1_train, y1_train = X1_train.to(device) , y1_train.to(device)
X1_test, y1_test = X1_test.to(device), y1_test.to(device)

epochs = 1000

for epoch in range(epochs):
    model2.train()

    y2_logits = model2(X1_train).squeeze()
    y2_preds = torch.round(torch.sigmoid(y2_logits)) # forward

    loss2 = loss2_fn(y2_logits, y1_train) # loss calculation

    optimizer2.zero_grad() # zero grad

    loss2.backward() # backpropagation

    optimizer2.step()

    model2.eval()
    with torch.inference_mode():
        y2_test_logits = model2(X1_test).squeeze()
        loss2_test = loss2_fn(y2_test_logits, y1_test)

    if epoch % 100 == 0:
        print(f"epoch {epoch} loss: {loss} \n test_loss: {loss2_test}") 



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plt_dec(model2,X1_train,y1_train)

plt.subplot(1,2,2)
plt.title("Test")
plt_dec(model2, X1_test,y1_test)
plt.show()