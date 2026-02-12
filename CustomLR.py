import numpy as np
from itertools import combinations_with_replacement



class LinearRegressionOLS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.beeta_ = None

    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis = 1)
        self.beeta_ = np.linalg.pinv(np.dot(x_train.T, x_train)) @ np.dot(x_train.T, y_train)
        self.intercept_ = self.beeta_[0]
        self.coef_ = self.beeta_[1:]

    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        return np.dot(x_test, self.beeta_)




class PolynomialLinearRegression:
    def __init__(self, degree):
        self.degree = degree
        self.intercept_ = None
        self.coef_ = None
        self.beta_ = None

    def create_polynomial_faetures(self, x):
        n_samples, n_features = x.shape
        self.combinations_list = []
        for deg in range(self.degree+1):
            self.combinations_list.extend(list(combinations_with_replacement(range(n_features), deg)))
        self.x_poly = np.ones((n_samples, len(self.combinations_list)))    
        
        for i, comb in enumerate(self.combinations_list):
            if (len(comb) > 0):
                self.x_poly[:, i] = np.prod(x[:, comb], axis = 1)
        return self.x_poly
        
    def fit(self, x_train, y_train):    
        self.x_train_poly = self.create_polynomial_faetures(x_train)
        self.beta_ = np.dot(np.linalg.pinv(self.x_train_poly), y_train) 
        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]

    def predict(self, x_test):
        self.x_test_poly = self.create_polynomial_faetures(x_test)
        return np.dot(self.x_test_poly, self.beta_)




class LinearRegressionBatchGD:
    
    def __init__(self,learning_rate=0.01,epochs=10):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            y_hat = np.dot(X_train,self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_der = -2 * np.dot((y_train - y_hat),X_train)/X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_