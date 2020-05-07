import dataAnalysis as da
import numpy as np
import math
from collections import Counter
import features as ft
import time

# basé sur l'algorithme ID3


class DecisionTreeClassifier:
    def __init__(self, X, Y, max_depth=2, threshold=2, split_criterion="entropy", gen_test="mean"):
        self.feature = X
        self.gold = Y
        self.classLabel = dict()
        self.labelClass = dict()
        self.max_depth = max_depth
        self.threshold = threshold
        self.split_criterion = split_criterion
        self.gen_test = gen_test

        # récupére les classes de chacun des labels
        occurLab = Counter(self.gold)
        id = 0
        for k, _ in occurLab.most_common():
            self.classLabel[k] = id
            self.labelClass[id] = k
            id += 1

        self.gold = np.array([self.classLabel[g] for g in self.gold])

    def setNewParam(self, newParam):
        self.max_depth = newParam[0]
        self.split_criterion = newParam[1]
        self.gen_test = newParam[2]

    # Fonction permettant de calculer l'entropie d'un ensemble
    def computeEntropy(self, Y):
        _, count = np.unique(Y, return_counts=True)
        proba = count / np.shape(Y)[0]
        entropy = proba * np.log2(proba)
        res = -np.sum(entropy)
        return res

    # Fonction permettant de calculer le degré d'impureté de l'ensemble Y
    def computeGiniImpurity(self, Y):
        _, count = np.unique(Y, return_counts=True)
        gini = np.power((count / np.shape(Y)[0]), 2)
        res = 1 - np.sum(gini)
        return res

    # Fonction calculant la moyenne de toute les features uniques
    def mean(self, X, nbfeature):
        res = np.zeros(nbfeature)
        for f in range(nbfeature):
            res[f] = np.mean(np.unique(X[:, f]))
        return res

    # Fonction calculant la médiane de toute les features uniques
    def median(self, X, nbfeature):
        res = np.zeros(nbfeature)
        for f in range(nbfeature):
            res[f] = np.median(np.unique(X[:, f]))
        return res

    # Fonction permettant de généré un ensemble de test dans le but de splitté au mieux les données
    # en deux sous_ensemble. Les labels doivent être le mieux répartie entre les deux sous_ensemble.
    # Par exemple, les NOUN d'un côté et le reste de l'autre

    def getBestTest(self, X, Y):

        _, nbFeature = np.shape(X)
        nbGold = np.shape(Y)[0]

        entropyY = 0

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
        bestYesLabel = None
        bestNoLabel = None
        
        bestYesIndex = None
        bestNoIndex = None
        
        if self.gen_test == "mean":
            test = self.mean(X, nbFeature)
        elif self.gen_test == "median":
            test = self.median(X, nbFeature)


        # On regarde chacune des features
        for f in range(nbFeature):
                
            yes_index = X[:, f] <= test[f]
            no_index = np.logical_not(yes_index)

            yes_label = Y[yes_index]
            no_label = Y[no_index]

            score = 0

            # Si le test permet de séparé les données
            if yes_label.size > 0 and no_label.size > 0:

                if self.split_criterion == "entropy":
                    # On calcule la quantité d'information contenue dans l'ensemble des données
                    # passant le test
                    entropyYes = self.computeEntropy(yes_label)
                    # De même dans l'ensemble des données ne passant pas le test
                    entropyNo = self.computeEntropy(no_label)
                    probaYes = np.shape(yes_label)[0] / nbGold
                    probaNo = np.shape(no_label)[0] / nbGold
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
                        bestTest = test[f]
                        bestYesLabel = yes_label
                        bestNoLabel = no_label
                        bestYesIndex = yes_index
                        bestNoIndex = no_index

                if self.split_criterion == "gini":
                    # On calcule le degrés d'impureté des deux ensembles no et yes
                    impurityLeft = self.computeGiniImpurity(yes_label)
                    impurityright = self.computeGiniImpurity(no_label)
                    probaYes = np.shape(yes_label)[0] / nbGold
                    probaNo = np.shape(no_label)[0] / nbGold
                    score = probaYes * impurityLeft + probaNo * impurityright

                    # On récupére le test permettant de minimiser l'impureté
                    if score < bestScore:
                        bestScore = score
                        bestFeature = f
                        bestTest = test[f]
                        bestYesLabel = yes_label
                        bestNoLabel = no_label
                        bestYesIndex = yes_index
                        bestNoIndex = no_index

        
        # On renvoie les données du test nous permettant d'obtenir le plus d'information
        # sur les données du noeud. On a splitté les données en deux sous_ensemble contenant
        # chacun plus d'information sur les données à prédire
        return bestFeature, bestTest, bestNoLabel, X[bestNoIndex], bestYesLabel, X[bestYesIndex]

    # Fonction permettant de généré l'arbre de décision grâce au donnée fournie au constructeur
    # de DecisionTree

    def fit(self):

        def build_tree(X, Y, max_depth):
            # test permettant de savoir si l'on a atteint une feuille
            # on s'arrête si l'on atteint la profondeur maximal ou que le nombre de données
            # du noeud passe sous la barre d'un certain seuil
            # ou que l'ensemble des labels ne contient qu'un seul type de labels,
            # (cela signifie que l'arbre à déjà réussi à discriminer ce labels pour cette branche)
            if np.shape(X)[0] <= self.threshold or np.shape(Y)[0] <= self.threshold or max_depth == 0 or np.shape(np.unique(Y))[0] == 1:
                return Counter(Y).most_common(1)[0][0]
            else:

                # on cherche le meilleur test permettant de séparé au mieux les données
                # de ce noeud
                bestFeature, bestTest, Yno, Xno, Yyes, Xyes = self.getBestTest(
                    X, Y)

                
                # Si on n'a pas réussi à séparé les données, on crée une feuille
                if (Yno is None
                    or Xno is None
                    or Yyes is None
                        or Xyes is None):
                    return Counter(Y).most_common(1)[0][0]

                # On construit la branche gauche de l'arbre avec les données ayant passé le test
                left = build_tree(Xyes, Yyes, (max_depth - 1))
                # On construit la branche droite de l'arbre avec les données n'ayant pas passé le test
                right = build_tree(Xno, Yno, (max_depth - 1))

            # On renvoit l'arbre
            return ((bestFeature, bestTest), left, right)

        return build_tree(self.feature, self.gold, self.max_depth)

    def predictSingleWord(self, node, X):
        if type(node) != tuple:
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
    # ensemble de test donnée + la matrice de confusion du modèle
    def modelScore(self, tree, X, Y):

        occurLab = Counter(Y)
        id = list(self.classLabel.values())[-1]
        for k, _ in occurLab.most_common():
            if not k in self.classLabel:
                self.classLabel[k] = id
                self.labelClass[id] = k
                id += 1

        nbLabel = len(self.classLabel.values())
        cm = np.zeros((nbLabel, nbLabel))

        Ypredict = self.predict(tree, X)
        score = 0
        for ygold, ypredict in zip(Y, Ypredict):
            cm[self.classLabel[ygold], self.classLabel[ypredict]] += 1
            if ygold == ypredict:
                score += 1
        return (score / len(Y)) * 100, cm

    def showTree(self, tree, s='', depth=0):
        for _ in range(depth-1):
            print('|    ', end="")
        if depth > 0:
            print('-', end="")
            print(s, end="")
            print('-> ', end="")
        if type(tree) == int:  # feuille
            print("class ", self.labelClass[tree])
        else:  # noeud interne
            feature, test = tree[0]
            feature = list(ft.features().keys())[feature]
            print(feature, " <= ", test, " ?")
            self.showTree(tree[1], s='Y', depth=depth+1)
            self.showTree(tree[2], s='N', depth=depth+1)
