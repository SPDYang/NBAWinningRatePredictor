import requests
import json
import numpy as np
import random
from Lasso import LassRegression


class dataManager:
	def __init__(self):
		self.trainingX = None
		self.trainingY = None
		self.testX = None
		self.testY = None
		self.features = []

	def handleNBAStat(self, split: float = 0.66):
		winpct = []
		dataset = []
		headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"}

		# Collect the data features
		url1 = "https://stats.nba.com/stats/leaguedashteamstats?" + \
		"Conference=&DateFrom=&DateTo=&Division=&GameScope=&Game" + \
		"Segment=&LastNGames=0&LeagueID=00&Location=&MeasureType" + \
		"=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceA" + \
		"djust=N&PerMode=PerGame&Period=0&PlayerExperience=&Play" + \
		"erPosition=&PlusMinus=N&Rank=N&Season=2010-11&SeasonSeg" + \
		"ment=&SeasonType=Regular+Season&ShotClockRange=&Starter" + \
		"Bench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
		response1 = requests.get(url1, headers = headers) 
		datatmp = response1.json()
		statHeaders = datatmp["resultSets"][0]["headers"]
		for i in range(6, 55):
			self.features.append(statHeaders[i])

		# Collect the raw data from NBA API
		for year in range(10, 19):
			url2 = "https://stats.nba.com/stats/leaguedashteamst" + \
			"ats?Conference=&DateFrom=&DateTo=&Division=&GameSco" + \
			"pe=&GameSegment=&LastNGames=0&LeagueID=00&Location=" + \
			"&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=" + \
			"&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&Pl" + \
			"ayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&" + \
			"Season=20" + str(year) + "-" + str(year+1) + "&Seas" + \
			"onSegment=&SeasonType=Regular+Season&ShotClockRange" + \
			"=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsD" + \
			"ivision="

			response2 = requests.get(url2, headers = headers)
			rawData = response2.json()
			teams = rawData["resultSets"][0]["rowSet"]
			for team in teams:
				teamHolder = []
				for i in range(5, 55):
					if i == 5:
						winpct.append(float(team[i]))
					else:
						teamHolder.append(float(team[i]))
				dataset.append(teamHolder)	

		trainingX = []
		trainingY = []
		testX = []
		testY = []
		rowLength = len(dataset)
		for row in range(rowLength):
			if random.random() < split:
				trainingX.append(dataset[row])
				trainingY.append(winpct[row])
			else:
				testX.append(dataset[row])
				testY.append(winpct[row])
		self.trainingX = np.array(trainingX)
		self.trainingY = np.array(trainingY)
		self.testX = np.array(testX)
		self.testY = np.array(testY)

		self.trainingX = (self.trainingX - self.trainingX.mean()) / self.trainingX.std()
		self.testX = (self.testX - self.testX.mean()) / self.testX.std()

		return self.trainingX, self.trainingY, self.testX, self.testY, self.features