import torch
import numpy as np 
from torch import nn as nn 
import matplotlib.pyplot as plt
import plotPredictions as pP
from pathlib import Path

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * X + bias

# print(y)

# Crate a training and test sets

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]
# print(len(X_train), len(X_test), len(y_train), len(y_test))



# plot_preditions(X_train, y_train, X_test, y_test)

# MODEL 

class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad = True,
                                                dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad = True,
                                            dtype = torch.float))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)

model_0 = linearRegressionModel()

print(list(model_0.parameters()))

print(model_0.state_dict())


# Tworzenie przewidywanych 
with torch.inference_mode():
    y_preds = model_0(X_test)
    # print(y_preds)

# print(y_test)

pP.plot_preditions(X_train, y_train, X_test, y_test, predictions=y_preds)


# Loss - roznica pomiedzy idealnym a naszym 
loss = nn.L1Loss()

print(loss)

#Optimizer - SGD - Stochastic Gradiant Descend
optimizer = torch.optim.SGD(params = model_0.parameters(), 
                            lr = 0.01)

epochs = 500

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):

    model_0.train() # ustawiamy requires_grad = true 

    y_pred = model_0(X_train) # forward - idziemy na koniec sieci

    loss1 = loss(y_pred, y_train) # obliczamy blad MSE w tym przypadku 
    print(model_0.state_dict())
    optimizer.zero_grad() # 

    loss1.backward()

    optimizer.step()

    model_0.eval() # ustawiamy requires_grad = false 

    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss(test_pred,y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss1)
        test_loss_values.append(test_loss)
        # print(f"Epoch: {epoch} | Loss: {loss1} | Test loss: {test_loss}")


for i in range(len(epoch_count)):
    print(f"Epoch: {epoch_count[i]} | Loss : {loss_values[i]} | Test Loss: {test_loss_values[i]}")

with torch.inference_mode():
    plt.plot(epoch_count,loss_values, label="Loss values")
    plt.plot(epoch_count,test_loss_values, label = "test loss values")
    plt.title("Test loss and loss values")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

pP.plot_preditions(X_train, y_train, X_test, y_test, predictions=test_pred)
print(weight,bias)

MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "Model_1_Regresja_Liniowa.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

torch.save(model_0.state_dict(),MODEL_SAVE_PATH)

print("\n")

loaded_model = linearRegressionModel()

loaded_model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))

print(model_0.state_dict())
print(loaded_model.state_dict())