# using gradient descent
# derivation of mathematics found in the report 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class LogisticRegressionModel(): 

    def __init__(self, X_train_processed, X_test_processed, y_train): 
        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed
        self.y_train = y_train
        self.params = None

    ## loss function formulation 

    def sigmoid(self, y): 
        return 1/(1+np.e**(-y))
    
    def loss(self, params):
        """
        more optimised loss function using vectorisation rather than for loops
        """

        u = self.sigmoid(np.dot(self.X_train_processed, params))  
        u = np.clip(u, 1e-15, 1 - 1e-15)

        y = self.y_train 

        lf = np.sum(-y*np.log(u) - (1-y)*np.log(1-u))

        return lf
    
    ## gradient calculation 

    def gradient_num(self, params, h): 
        """
        find gradient with respect to each parameter using central difference method 
        (update this function to support the new loss function)
        """
        grad = []

        for index in range(len(params)): 

            modified1 = params.copy()
            modified1[index] = modified1[index] + h

            modified2 = params.copy()
            modified2[index] = modified2[index] - h

            grad.append(
                (self.loss(modified1) - self.loss(modified2)) / (2*h)
            )


        # dfdt1 = (self.f(t1 + h, t2, t3, t4, t5, t6, t7, t8) - self.f(t1 - h, t2, t3, t4, t5, t6, t7, t8)) / (2 * h)
        # dfdt2 = (self.f(t1, t2 + h, t3, t4, t5, t6, t7, t8) - self.f(t1, t2 - h, t3, t4, t5, t6, t7, t8)) / (2 * h)
        # dfdt3 = (self.f(t1, t2, t3 + h, t4, t5, t6, t7, t8) - self.f(t1, t2, t3 - h, t4, t5, t6, t7, t8)) / (2 * h)
        # dfdt4 = (self.f(t1, t2, t3, t4 + h, t5, t6, t7, t8) - self.f(t1, t2, t3, t4 - h, t5, t6, t7, t8)) / (2 * h)
        # dfdt5 = (self.f(t1, t2, t3, t4, t5 + h, t6, t7, t8) - self.f(t1, t2, t3, t4, t5 - h, t6, t7, t8)) / (2 * h)
        # dfdt6 = (self.f(t1, t2, t3, t4, t5, t6 + h, t7, t8) - self.f(t1, t2, t3, t4, t5, t6 - h, t7, t8)) / (2 * h)
        # dfdt7 = (self.f(t1, t2, t3, t4, t5, t6, t7 + h, t8) - self.f(t1, t2, t3, t4, t5, t6, t7 - h, t8)) / (2 * h)
        # dfdt8 = (self.f(t1, t2, t3, t4, t5, t6, t7, t8 + h) - self.f(t1, t2, t3, t4, t5, t6, t7, t8 - h)) / (2 * h)

        return np.array(grad)
    
    def gradient_optimised(self, params):
        """
        calculated gradient analytically 
        """
        u = self.sigmoid(np.dot(self.X_train_processed, params))        
        gradient = self.X_train_processed.T @ (u - self.y_train)
        return gradient

    ## find parameters using: gradient descent 
    
    def gradient_descent(self, start, lr, n_iter, method):
        """
        find optimal params using gradient descent
        """
        cost = []
        if method == "numerical":
            print("gradient descent by using numerical gradient ...")
            x = np.array(start)
            h = 0.01
            for _ in range(n_iter):
                cost.append(self.loss(x))
                grad = self.gradient_num(x, h)
                x = x - lr * grad
            self.params = x
        else: 
            print("gradient descent by using analytical gradient ...")
            x = np.array(start)
            for _ in range(n_iter):
                cost.append(self.loss(x))
                grad = self.gradient_optimised(x)
                if np.abs(grad.T @ grad) < 0.01:
                    print("found params first")
                    break
                else:
                    x = x - lr * grad
            self.params = x
        
        ## add convergence graph 
        ## y-axis value of the loss function
        ## no. of iterations
        ## legend: learning rate, numerical vs different

        plt.plot(cost)
        plt.xlabel("no of iterations")
        plt.ylabel("cost function")
        plt.title("cost function vs iterations")
        plt.grid(True)
        plt.show()

        return self.params


    
    ## find parameters using: newton's method 
        
    def newton_raphson(self, start, n_iter):
        """
        approach find the jacobian of the gradient which is the hessian by treating the gradient 
        as a vector of functions 
        """
        print("running newton's method")

        h = 0.01 # define accuract of numerical differentiation

        cost = []

        result = start 

        for _ in range(n_iter): 

            cost.append(self.loss(result))

            result = result - np.linalg.inv(self.jacobian(result, h)) @ self.gradient_optimised(result)
        
        print("params found!")

        self.params = result

        plt.plot(cost)
        plt.xlabel("no of iterations")
        plt.ylabel("cost function")
        plt.title("cost function vs iterations")
        plt.grid(True)
        plt.show()

        return self.params

    def jacobian(self, params, h):
        """
        the jacobian in netwon-raphson is the gradient of each component in the gradient vector
        """

        # jac = []

        # grad_vector = self.gradient_optimised(params) # the gradient vector 

        # for component in grad_vector: 

        #     component_grad = []

        #     for index in range(len(params)): 

        grad = []

        for index in range(len(params)): 

            modified1 = params.copy()
            modified1[index] = modified1[index] + h

            modified2 = params.copy()
            modified2[index] = modified2[index] - h

            grad.append(
                (self.gradient_optimised(modified1) - self.gradient_optimised(modified2)) / (2*h)
            )
        
        return grad
    
    # predictions

    def predict(self): 
        """
        find predictions 
        """
        threshold = 0.5

        predictions = []

        self.params

        for array in self.X_test_processed: 
            prob = self.sigmoid(np.dot(self.params, array))

            if prob > threshold: 
                predictions.append(1)
            else: 
                predictions.append(0)

        return np.array(predictions)

if __name__ == "__main__":

    from preprocessor import Preprocessor

    FILE_PATH = '/Users/thananpornsethjinda/Desktop/pulsar-pred/data/pulsar_data.csv'

    df = pd.read_csv(FILE_PATH)

    X_train_processed, X_test_processed, y_train, y_test = Preprocessor(df).main()

    start = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    lr = LogisticRegressionModel(X_train_processed, X_test_processed, y_train)

    # print(len(lr.jacobian(start, h=0.01)[0]))

    print(lr.newton_raphson(start=start, n_iter=100))

    print(lr.params)



    # print(lr.gradient_descent(start=start, lr=0.01, n_iter=100, method="hehe"))

    y_pred = lr.predict()

    from sklearn.metrics import roc_auc_score, confusion_matrix

    print(roc_auc_score(y_true=y_test, y_score=y_pred))

    print(confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true'))


## aims by the end of this study session: 
# - optimise this function 
# - try to get the optimal parameters 




    

