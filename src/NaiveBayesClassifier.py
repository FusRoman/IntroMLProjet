import dataAnalysis as da
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np
import features as ft
import math

pds.set_option('display.max_colwidth', -1)
da.init()
trainPartut, testPartut = da.listeCorpus["partut"].trainDataSet, da.listeCorpus["partut"].testDataSet
Xtrain, Ytrain = ft.buildFeature(trainPartut)
Xtest, Ytest = ft.buildFeature(testPartut)

# ^ To be removed for the notebook
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# Pour l'instant on sépare par mots mais je suis pas sûr de moi 
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
    count = 0
    avg = 0
    for x in column:
        count += 1
        avg += x
    avg /= count
    
    # Ecart type
    sum = 0
    for x in column:
        sum += (x - avg) ** 2
    stdev = math.sqrt(sum / count)
    
    return {"count": count, "avg": avg, "stdev": stdev}

# Pour chaque colonne d'un ensemble de données, calcule le compte, 
# la moyenne et l'écart type
def summarize(dataset):
    return [summarizeColumn(column) for column in zip(*dataset)]

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-((x-mean)**2 / (2 * stdev**2)))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    
        
class NaiveBayesClassifier:
    def __init__(self, X, Y):
        self.data = separateByClass(X, Y)
        
    def fit(self):
        self.summaries = dict()
        for _class, rows in self.data.items():
            self.summaries[_class] = summarize(rows)
            
        for _class, rows in self.summaries.items():
            print(_class)
            for r in rows:
                print("\t", r)
    
    def predict(self, x):
        # Calcul des probabilités pour chaque classe
        totalRows = sum([summaries[label][0]["count"] for label in self.summaries])
        probabilities = dict()
        for _class, classSummaries in self.summaries.items():
            probabilities[_class] = classSummaries[0]["count"] / float(totalRows)
            for i in range(len(classSummaries)):
                columnSummary = classSummaries[i]
                mean = columnSummary["mean"]
                stdev = columnSummary["stdev"]
                probabilities[_class] *= calculateProbabilities(x[i], mean, stdev)
        
        # argmax
        result = probabilities.keys()[0]
        for _class, p in probabilities.items():
            if p > probabilities[result]:
                result = _class
        return result

classifier = NaiveBayesClassifier(Xtrain, Ytrain)
classifier.fit()

def test(clf, X, Y):
    
    for i in range(len(X)):
        vector = X[i]
        gold = Y[i]