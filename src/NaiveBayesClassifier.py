import dataAnalysis as da
import matplotlib.pyplot as plt
import numpy as np
import features as ft
import math


# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# Renvoie un dictionnaire dont les entrées sont les classes,
# et les valeurs les données de cette classe
def separateByClass(X, Y):
    separated = dict()
    for i in range(len(X)):
        gold = Y[i]
        if gold not in separated:
            separated[gold] = []
        separated[gold].append(X[i])
                
    return separated

# Fonction auxiliaire de summarize
def summarizeColumn(column):
    # Compte et moyenne
    count = len(column)
    avg = 0
    for x in column:
        avg += x / count
    
    # Ecart type
    sum = 0
    for x in column:# Pour l'instant on sépare par mots mais je suis pas sûr de moi
        sum += (x - avg) ** 2
    stdev = math.sqrt(sum / count)
    
    
    return {"count": count, "avg": avg, "stdev": stdev}

# Pour chaque colonne d'un ensemble de données, calcule le compte,
# la moyenne et l'écart type
def summarize(dataset):
    return [summarizeColumn(column) for column in zip(*dataset)]

def calculateProbability(x, mean, stdev, sigma):
    s = stdev
    if stdev == 0:
        s = sigma
    exponent = math.exp(-((x-mean)**2 / (2 * s**2)))
    return (1 / (math.sqrt(2 * math.pi) * s)) * exponent
    
        
class NaiveBayesClassifier:
    def __init__(self, X, Y, sigma = 0.0000000000000001):
        self.data = separateByClass(X, Y)
        self.sigma = sigma
        
    def fit(self):
        self.summaries = dict()
        for _class, rows in self.data.items():
            self.summaries[_class] = summarize(rows)
    
    def predictSingleWord(self, x):
        # Calcul des probabilités pour chaque classe
        totalRows = sum([self.summaries[label][0]["count"] for label in self.summaries])
        probabilities = dict()
        for _class, classSummaries in self.summaries.items():
            probabilities[_class] = classSummaries[0]["count"] / float(totalRows)
            for i in range(len(classSummaries)):
                columnSummary = classSummaries[i]
                avg = columnSummary["avg"]
                stdev = columnSummary["stdev"]
                probabilities[_class] *= calculateProbability(x[i], avg, stdev, self.sigma)
        
        # argmax
        result = list(probabilities.keys())[0]
        for _class, p in probabilities.items():
            if p > probabilities[result]:
                result = _class
        return result
    
    def predict(self, X):
        res = []
        for feature in X:
            res.append(self.predictSingleWord(feature))
        return res
    
    def modelScore(self, X, Y):
    
        allLabels = self.summaries.keys()
        classLabel = dict()
        id = 0
        for lab in allLabels:
            classLabel[lab] = id
            id += 1
        
        nbLabel = len(allLabels)
        cm = np.zeros( (nbLabel, nbLabel) )
        
        Ypredict = self.predict(X)
        score = 0
        for ygold, ypredict in zip(Y, Ypredict):
            cm[classLabel[ygold], classLabel[ypredict]] += 1
            if ygold == ypredict:
                score += 1
        return ((score / len(Y)) * 100, cm)
