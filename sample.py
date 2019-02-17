import numpy as np
from dataManager import dataManager
from Lasso import LassRegression


if __name__ == "__main__":
	dataset = dataManager()
	trainingX, trainingY, testX, testY, features = dataset.handleNBAStat()
	trail1 = LassRegression(alpha = 0.001, iterations = 1000)
	trail1.fit(trainingX, trainingY)
	# print(trail1.intercept)
	# print(trail1.coefficient)
	print("\n+++++++++++ THE DATASET HAS THE FOLLOWING FEATURES: +++++++++++")
	print(*features)
	print("\nTARGET: W_PCT")
	print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")	
	print(trail1.makePrediction(testX[0]))
	print(testY[0])
	print(trail1.score(trainingX, trainingY))
	print(trail1.score(testX, testY))

	
