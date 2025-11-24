# using gradient descent
# derivation of mathematics found in the report 
import numpy as np
import pandas as pd 
from preprocessor import Preprocessor


class LogisticRegressionModel(): 

    def __init__(self, X_train_processed, y_train): 
        self.X_train_processed = X_train_processed
        self.y_train = y_train
        self.params = None

    def sigmoid(self, y): 
        return 1/(1+np.e**(-y))
    
    def gradient_descent(self, start, lr, n_iter):
        """
        find optimal params using gradient descent
        """
        x = np.array(start)
        h = 0.01
        for _ in range(n_iter):
            grad = self.gradient(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], h)
            x = x - lr * grad
        self.params = x
        return self.params
    
    def gradient(self, t1, t2, t3, t4, t5, t6, t7, t8, h): 
        """
        find gradient with respect to each parameter using central difference method 
        """

        dfdt1 = (self.f(t1 + h, t2, t3, t4, t5, t6, t7, t8) - self.f(t1 - h, t2, t3, t4, t5, t6, t7, t8)) / (2 * h)
        dfdt2 = (self.f(t1, t2 + h, t3, t4, t5, t6, t7, t8) - self.f(t1, t2 - h, t3, t4, t5, t6, t7, t8)) / (2 * h)
        dfdt3 = (self.f(t1, t2, t3 + h, t4, t5, t6, t7, t8) - self.f(t1, t2, t3 - h, t4, t5, t6, t7, t8)) / (2 * h)
        dfdt4 = (self.f(t1, t2, t3, t4 + h, t5, t6, t7, t8) - self.f(t1, t2, t3, t4 - h, t5, t6, t7, t8)) / (2 * h)
        dfdt5 = (self.f(t1, t2, t3, t4, t5 + h, t6, t7, t8) - self.f(t1, t2, t3, t4, t5 - h, t6, t7, t8)) / (2 * h)
        dfdt6 = (self.f(t1, t2, t3, t4, t5, t6 + h, t7, t8) - self.f(t1, t2, t3, t4, t5, t6 - h, t7, t8)) / (2 * h)
        dfdt7 = (self.f(t1, t2, t3, t4, t5, t6, t7 + h, t8) - self.f(t1, t2, t3, t4, t5, t6, t7 - h, t8)) / (2 * h)
        dfdt8 = (self.f(t1, t2, t3, t4, t5, t6, t7, t8 + h) - self.f(t1, t2, t3, t4, t5, t6, t7, t8 - h)) / (2 * h)

        return np.array([dfdt1, dfdt2, dfdt3, dfdt4, dfdt5, dfdt6, dfdt7, dfdt8])
    
    
    def f(self, t1, t2, t3, t4, t5, t6, t7, t8):

        loss_func = []

        z = np.array([t1, t2, t3, t4, t5, t6, t7, t8])

        for index in range(len(self.X_train_processed)): 

            h = self.sigmoid(np.dot(z, self.X_train_processed[index]))

            h = np.clip(h, 1e-15, 1 - 1e-15)

            y = np.array(self.y_train)[index]

            loss_func.append(-np.log(h**y*(1-h)**(1-y)))

        return np.sum(loss_func)

    def predict(self): 
        """
        find predictions 
        """

        threshold = 0.5

        predictions = []

        self.params

        for array in self.X_train_processed: 
            prob = self.sigmoid(np.dot(self.params, array))

            if prob > threshold: 
                predictions.append(1)
            else: 
                predictions.append(0)

        return np.array(predictions)

if __name__ == "__main__":

    FILE_PATH = '/Users/thananpornsethjinda/Desktop/pulsar-pred/data/pulsar_data.csv'

    df = pd.read_csv(FILE_PATH)

    X_train_processed, X_test_processed, y_train, y_test = Preprocessor(df).main()

    start = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    lr = LogisticRegressionModel(X_train_processed, y_train)

    print(lr.gradient_descent(start=start, lr=0.01, n_iter=100))






    

