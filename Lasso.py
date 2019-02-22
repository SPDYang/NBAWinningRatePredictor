import pandas as pd 
import numpy as np


class LassRegression:
	def __init__(self, alpha: float = 1.0, iterations: int = 1000, fitIntercept: bool = True):
		self.alpha: float = alpha
		self.iterations: float = iterations
		self.fitIntercept: bool = fitIntercept
		self.coefficient: float = None
		self.intercept: float = None

	def softThresholding(self, p: float, lamb: float, z: float):
		theta: float = None
		if p > 0.0 and lamb < abs(p):
			theta = (p - lamb) / z
		elif p < 0.0 and lamb < abs(p):
			theta = (p + lamb) / z
		else:
			theta = 0.0
		return theta

	def fit(self, X: np.ndarray, Y: np.ndarray):
		n = X.shape[0]
		
		if self.fitIntercept == True:
			X = np.column_stack((np.ones(n), X))

		theta = np.zeros(X.shape[1])

		if self.fitIntercept == True:
			theta[0] = np.sum(Y - np.dot(X[:, 1:], theta[1:])) / n
		
		m = theta.shape[0]
		for iteration in range(self.iterations):
			begin = 1 if self.fitIntercept else 0
			for j in range(begin, m):
				thetatmp = theta.copy()
				thetatmp[j] = 0.0
				loss = Y - np.dot(X, thetatmp)
				p = np.dot(X[:, j], loss)
				lamb = self.alpha * n
				z = np.sum(X[:, j] ** 2)
				theta[j] = self.softThresholding(p, lamb, z)
				if self.fitIntercept == True:
					theta[0] = np.sum(Y - np.dot(X[:, 1:], theta[1:])) / n

		if self.fitIntercept == True:
			self.coefficient = theta[1:]
			self.intercept = theta[0]
		else:
			self.coefficient = theta

	def makePrediction(self, X: np.ndarray):
		newY = np.dot(X, self.coefficient)
		if self.fitIntercept:
			newY += self.intercept
		return newY

	def score(self, X: np.ndarray, testY: np.ndarray):
		predictY = self.makePrediction(X)
		sst = np.sum((testY - testY.mean()) ** 2)
		ssr = np.sum((testY - predictY) ** 2)
		return 1 - ssr / sst
