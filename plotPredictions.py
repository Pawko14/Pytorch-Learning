import matplotlib.pyplot as plt

def plot_preditions(train_data, train_labels, test_data, test_labels, predictions = None):
    #Plaszczyzna 
    plt.figure(figsize=(10,7))

    # Wykres X i Y
    plt.scatter(train_data, train_labels, c = "b", s = 4 , label = "Training data")

    plt.scatter(test_data, test_labels, c = "g" , s =4 , label = "Test Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r" , s = 4 , label = "Predictions")

    plt.legend(prop = {"size": 14});
    plt.show()
