import dataAnalysis as da
import numpy as np
import math
from collections import Counter
from itertools import product
import hashlib
import copy
import features as ft

# basé sur l'algorithme ID3
class DecisionTreeClassifier:
    def __init__(self, X, Y, max_depth=2, threshold=2, split_criterion = "entropy"):
        self.feature = X
        self.gold = Y
        self.classLabel = dict()
        self.labelClass = dict()
        self.max_depth = max_depth
        self.threshold = threshold
        self.split_criterion = split_criterion

        # récupére les classes de chacun des labels
        occurLab = Counter(self.gold)
        id = 0
        for k, _ in occurLab.most_common():
            self.classLabel[k] = id
            self.labelClass[id] = k
            id += 1

        self.gold = [self.classLabel[g] for g in self.gold]


    # Fonction permettant de calculer l'entropie d'un ensemble
    def computeEntropy(self, Y):
        res = 1
        lenY = len(Y)
        occurGold = Counter(Y)
        for _, occur in occurGold.items():
            prob = occur / lenY
            res -= prob * math.log2(prob)
        return res
    
    # Fonction permettant de calculer le degé d'impureté de l'ensemble Y
    def computeGiniImpurity(self, Y):
        res = 0
        lenY = len(Y)
        occurGold = Counter(Y)
        for _, occur in occurGold.items():
            res += (occur / lenY) ** 2
        return 1 - res

    # Fonction permettant de généré un ensemble de test dans le but de splitté au mieux les données
    # en deux sous_ensemble. Les labels doivent être le mieux répartie entre les deux sous_ensemble.
    # Par exemple, les NOUN d'un côté et le reste de l'autre

    def getBestTest(self, X, Y, ignore_feature):

        entropyY = 0
        nbGold = len(Y)
        # Calcule de l'entropie de l'ensemble contenant les données actuelle du noeud.
        # Plus l'entropie est faible, plus l'on a d'information sur l'ensemble actuelle
        if self.split_criterion == "entropy":
            entropyY = self.computeEntropy(Y)

        # Les variables permettant de garder les données du meilleurs test
        bestFeature = 0
        bestScore = 100000000
        
        if self.split_criterion == "entropy":
            bestScore = -1000000000
            
        bestTest = 0
        bestYesLabel = []
        bestYesSample = []
        bestNoLabel = []
        bestNoSamples = []

        # On regarde chacune des features
        for f in range(len(X[0])):

            setFeature = set()

            for otherFeature in X:
                setFeature.add(otherFeature[f])

            # On génère un test en prenant la moyennes sur les valeurs uniques de la features f
            averageFeature = sum(setFeature) / len(setFeature)
            if (f, averageFeature) not in ignore_feature:
                no_label = []
                yes_label = []

                yes_samples = []
                no_samples = []

                # Pour toutes les données
                for samplesIdx in range(len(X)):
                    # On splitte les données en deux sous_ensembles
                    if X[samplesIdx][f] <= averageFeature:
                        # L'ensemble des données qui passe le test
                        yes_label.append(Y[samplesIdx])
                        yes_samples.append(X[samplesIdx])
                    else:
                        # L'ensemble des données qui échoue au test
                        no_label.append(Y[samplesIdx])
                        no_samples.append(X[samplesIdx])

                score = 0
                
                # Si le test permet de séparé les données
                if yes_label and no_label and yes_samples and no_samples:

                    if self.split_criterion == "entropy":
                        # On calcule la quantité d'information contenue dans l'ensemble des données
                        # passant le test
                        entropyYes = self.computeEntropy(yes_label)
                        # De même dans l'ensemble des données ne passant pas le test
                        entropyNo = self.computeEntropy(no_label)
                        probaYes = len(yes_label) / nbGold
                        probaNo = len(no_label) / nbGold
                        # On calcule l'entropie conditionnelle H(S|test), c'est à dire la quantité
                        # d'information contenue dans l'ensemble des données du noeud sachant qu'un
                        # test a été effectué
                        Ichild = probaYes * entropyYes + probaNo * entropyNo
                        # On calcule le gain d'information obtenue par le test
                        score = entropyY - Ichild
                        
                        # On récupére le test permettant de maximiser le gain d'information
                        if score > bestScore:
                            bestScore = score
                            bestFeature = f
                            bestTest = averageFeature
                            bestYesLabel = yes_label
                            bestYesSample = yes_samples
                            bestNoLabel = no_label
                            bestNoSamples = no_samples
                        
                        
                    
                    if self.split_criterion == "gini":
                        # On calcule le degrés d'impureté des deux ensembles no et yes
                        impurityLeft = self.computeGiniImpurity(yes_label)
                        impurityright = self.computeGiniImpurity(no_label)
                        probaYes = len(yes_label) / nbGold
                        probaNo = len(no_label) / nbGold
                        score = probaYes * impurityLeft + probaNo * impurityright
                        
                        # On récupére le test permettant de minimiser l'impureté 
                        if score < bestScore:
                            bestScore = score
                            bestFeature = f
                            bestTest = averageFeature
                            bestYesLabel = yes_label
                            bestYesSample = yes_samples
                            bestNoLabel = no_label
                            bestNoSamples = no_samples

        # On renvoie les données du test nous permettant d'obtenir le plus d'information
        # sur les données du noeud. On a splitté les données en deux sous_ensemble contenant
        # chacun plus d'information sur les données à prédire
        return bestFeature, bestTest, bestNoLabel, bestNoSamples, bestYesLabel, bestYesSample

    # Fonction permettant de généré l'arbre de décision grâce au donnée fournie au constructeur
    # de DecisionTree

    def fit(self):

        def build_tree(X, Y, max_depth, ignore_feature=set()):

            # test permettant de savoir si l'on a atteint une feuille
            # on s'arrête si l'on atteint la profondeur maximal ou que le nombre de données
            # du noeud passe sous la barre d'un certain seuil
            if len(X) <= self.threshold or len(Y) <= self.threshold or max_depth == 0:
                return Counter(Y).most_common(1)[0][0]
            else:

                # on cherche le meilleur test permettant de séparé au mieux les données
                # de ce noeud
                bestFeature, bestTest, Yno, Xno, Yyes, Xyes = self.getBestTest(
                    X, Y, ignore_feature)

                # Si on n'a pas réussi à séparé les données, on créée une feuille
                if not Yno or not Xno or not Yyes or not Xyes:
                    return Counter(Y).most_common(1)[0][0]

                # On ajoute le test de ce noeud aux tests à ignoré pour les
                # noeud enfant
                ignore_feature = copy.deepcopy(ignore_feature)
                ignore_feature.add((bestFeature, bestTest))

                # On construit la branche gauche de l'arbre avec les données ayant passé le test
                left = build_tree(Xyes, Yyes, (max_depth - 1),
                                  ignore_feature)
                # On construit la branche droite de l'arbre avec les données n'ayant pas passé le test
                right = build_tree(Xno, Yno, (max_depth - 1),
                                   ignore_feature)

            # On renvoit l'arbre
            return ((bestFeature, bestTest), left, right)

        return build_tree(self.feature, self.gold, self.max_depth)

    def predictSingleWord(self, node, X):
        if type(node) == int:
            return self.labelClass[node]
        else:
            feature, test = node[0]
            if X[feature] <= test:
                return self.predictSingleWord(node[1], X)
            else:
                return self.predictSingleWord(node[2], X)

    def predict(self, node, X):
        res = []
        for feature in X:
            res.append(self.predictSingleWord(node, feature))
        return res

    # Fonction renvoyant le pourcentage de prédiction correcte pour un arbre et un 
    # ensemble de test donnée
    def modelScore(self, tree, X, Y):
        Ypredict = self.predict(tree, X)
        score = 0
        for ygold, ypredic in zip(Y, Ypredict):
            if ygold == ypredic:
                score += 1
        return (score / len(Y)) * 100
    

    def showTree(self, tree, s='', depth=0):
        for _ in range(depth-1):
            print('|    ', end="")
        if depth > 0:
            print('-', end="")
            print(s, end="")
            print('-> ', end="")
        if type(tree) == int: # leaf
            print("class ", self.labelClass[tree])
        else: # internal node
            feature, test = tree[0]
            feature = list(ft.features().keys())[feature]
            print(feature, " <= ", test, " ?")
            self.showTree(tree[1], s='Y', depth=depth+1)
            self.showTree(tree[2], s='N', depth=depth+1)


