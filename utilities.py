import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_data():
    train_dataset = h5py.File('C:/Users/issla/Documents/GitHub/cat-vs-dog/trainset.hdf5', "r")
    x_train = np.array(train_dataset["X_train"][:])  # your train set features
    y_train = np.array(train_dataset["Y_train"][:])  # your train set labels

    test_dataset = h5py.File('C:/Users/issla/Documents/GitHub/cat-vs-dog/testset.hdf5', "r")
    x_test = np.array(test_dataset["X_test"][:])  # your train set features
    y_test = np.array(test_dataset["Y_test"][:])  # your train set labels

    return x_train, y_train, x_test, y_test

def initialisation(x):
    w = np.random.randn(x.shape[1], 1)
    b = np.random.randn(1)
    return w, b


def model(x, w, b):
    z = x.dot(w) + b
    # print(z.min())
    a = 1 / (1 + np.exp(-z))
    return a


def log_loss(a, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(a + epsilon) - (1 - y) * np.log(1 - a + epsilon))


def gradients(a, x, y):
    dw = 1 / len(y) * np.dot(x.T, a - y)
    db = 1 / len(y) * np.sum(a - y)
    return dw, db


def update(dw, db, w, b, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b


def predict(x, w, b):
    a = model(x, w, b)
    print(a)
    return a >= 0.5


def artificial_neuron(x_train, y_train, x_test, y_test, learning_rate=0.1, n_iter=100):
    # initialisation w, b
    w, b = initialisation(x_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        a = model(x_train, w, b)

        if i % 10 == 0:
            # Train
            train_loss.append(log_loss(a, y_train))
            y_pred = predict(x_train, w, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Test
            a_test = model(x_test, w, b)
            test_loss.append(log_loss(a_test, y_test))
            y_pred = predict(x_test, w, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise a jour
        dw, db = gradients(a, x_train, y_train)
        w, b = update(dw, db, w, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.legend()
    plt.show()

    return w, b
