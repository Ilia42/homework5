##########################################################
#           PERCEPTRON ALGORITHM FROM SCRATCH            #
##########################################################

# import packages
import numpy as np
import pandas as pd

# class for Perceptron algorithm 23
class PerceptronAlgorithm(object):
    def __init__(self, eta, max_epochs, threshold):
        self.eta = eta
        self.max_epochs = max_epochs
        self.threshold = threshold

    def get_weights(self, n):
        self.w = np.random.rand(n)
        self.b = np.random.rand(1)

    def input_net(self, x):
        net = np.dot(x, self.w) + self.b
        return net

    def f(self, net):
        return np.where(net >= 0.5, 1, 0)

    def predict(self, x):
        y_pred = self.f(self.input_net(x))
        return y_pred

    def loss_fn(self, y, y_pred):
        loss = (y - y_pred)
        return loss        

    def fit(self, x_train, y_train):
        n = x_train.shape[0]
        E = 2 * self.threshold
        count = 0
        self.get_weights(x_train.shape[1])
        cost = list()
        
        while (E >= self.threshold and count <= self.max_epochs + 1):
            E = 0
            for i in range(n):
                xi = x_train[i, :]
                yi = y_train[i]
                y_hat = self.predict(xi)
                error = self.loss_fn(yi, y_hat)
                E = E + error**2
                dE_dW = -error * xi
                dE_db = -error
                self.w = self.w - self.eta * dE_dW
                self.b = self.b - self.eta * dE_db
                
            count = count + 1
            E = 1/2 * (E/n)
            cost.append(E)            
            print('Epoch ', count, ' ===> error = ', E, '... \n')
            
        self.n_epochs = count
        self.loss = E
        self.cost_ = cost
        
        return self
    
    def test(self, x_test, y_test):
        n = x_test.shape[0]
        self.accuracy = 0 
        y_pred = list()
        for i in range(n):
            xi = x_test[i, :]
            yi = y_test[i]
            y_pred.append(self.predict(xi))
            if y_pred[i] == yi:
                self.accuracy = self.accuracy + 1
        
        self.accuracy = 100 * round(self.accuracy/n, 5)
        return y_pred
    
    # Новый метод для отображения модели
    def print_model(self):
        print("Perceptron model:")
        weights_str = " + ".join([f"{self.w[i]:.4f} * x{i+1}" for i in range(len(self.w))])
        print(f"y = {self.b[0]:.4f} + {weights_str}")
