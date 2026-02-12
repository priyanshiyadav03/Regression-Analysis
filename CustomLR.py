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
	def __init__(self, learning_rate = 0.01, epochs = 10):
		self.coef_ = None
		self.intercept_ = None
		self.learning_rate = learning_rate
		self.epochs = epochs

	def fit(self, x_train, y_train):
		self.coef_ = np.ones(x_train.shape[0])
		self.intercept_ = 0
		for i in range(self.epochs):
			self.subtr = y_train - (self.coef_ + np.dot(x_train, sefl.coef_))
			self.intercept_slope = -2 * np.mean(self.subtr)
			self.coef_slope = (-2/x_train.shape[0]) * np.dot(self.subtr, x_train) 
			self.coef_ = self.coef_ - (learning_rate * self.coef_slope)
			self.intercept_ = self.intercept_ - (learning_rate * self.intercept_slope)

	def transform(self, x_test):
		y_test = self.intercept_ + np.dot(x_train, self.coef_)
		return y_test