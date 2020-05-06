import dataAnalysis as da
import numpy as np
import math
from collections import Counter
from statistics import median


# Tableau de feature dont les arguments sont :
# s => the sentence
# i => the index of the word
# t => the transform to apply (hash for exemple)
def features():
    return {
        'word': lambda s, i, t: t(s[i]),

        'pos': lambda s, i, t: i-2,
        'first': lambda s, i, t: i == 2,
        'last': lambda s, i, t: i == len(s) - 3,
        'len': lambda s, i, t: len(s[i]),

        'startUpper': lambda s, i, t: s[i][0].isupper(),
        'hasUpper': lambda s, i, t: any(c.isupper() for c in s[i]),
        'allUpper': lambda s, i, t: s[i].isupper(),
        #'all-alpha': lambda s, i, t: s[i].isalpha(),
        # 'has-digit': lambda s, i, t: any(c.isdigit() for c in s[i]),
        'all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i]),

        'w-1': lambda s, i, t: t(s[i-1]),
        'w-2': lambda s, i, t: t(s[i-2]),
        'w+1': lambda s, i, t: t(s[i+1]),
        'w+2': lambda s, i, t: t(s[i+2]),

        #'w-1-all-alpha': lambda s, i, t: s[i-1].isalpha(),
        #'w-1-all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i-1]),
        #'w+1-all-alpha': lambda s, i, t: s[i+1].isalpha(),
        #'w+1-all-special': lambda s, i, t: not any(c.isalpha() or c.isdigit() for c in s[i+1]),

        'prefix-1': lambda s, i, t: t(s[i][:1]),
        'prefix-2': lambda s, i, t: t(s[i][:2]),
        'prefix-3': lambda s, i, t: t(s[i][:3]),

        'suffix-1': lambda s, i, t: t(s[i][-1:]),
        'suffix-2': lambda s, i, t: t(s[i][-2:]),
        'suffix-3': lambda s, i, t: t(s[i][-3:]),
        
        #'voyel-ratio':   lambda s, i, t: (lambda w:sum(w.count(v) for v in "aeiouy") / len(w) )(s[i].lower())
    }

# permet d'extraire les features d'une phrases
def extract_features(s, i, transform=lambda x: x):
    return tuple(f(s, i, transform) for f in features().values())


def preprocessSentence(dataset):
    begin_s = "<s>"
    end_s = "</s>"
    for d in dataset.data:
        d.sentence = [begin_s, begin_s] + d.sentence + [end_s, end_s]
        d.labels = [begin_s, begin_s] + d.labels + [end_s, end_s]

# extrait les features de tous le corpus train
def buildFeature(dataset):
    preprocessSentence(dataset)
    X = []
    Y = []
    for d in dataset.data:
        X += [extract_features(d.sentence, i, transform=hash)
              for i in range(2, len(d.sentence) - 3)]
        Y += d.labels[2:-3]
    return np.array(X), np.array(Y)


# basé sur l'algorithme ID3
class DecisionTreeClassifier:
    def __init__(self, X, Y, max_depth=2, threshold=2, split_criterion = "entropy", gen_test="mean"):
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

        self.gold = [self.classLabel[g] for g in self.gold]


    def setNewParam(self, newParam):
        self.max_depth = newParam[0]
        self.split_criterion = newParam[1]
        self.gen_test = newParam[2]

    # Fonction permettant de calculer l'entropie d'un ensemble
    def computeEntropy(self, Y):
        res = 1
        lenY = np.shape(Y)[0]
        occurGold = Counter(Y)
        for _, occur in occurGold.items():
            prob = occur / lenY
            res -= prob * math.log2(prob)
        return res
    
    # Fonction permettant de calculer le degé d'impureté de l'ensemble Y
    def computeGiniImpurity(self, Y):
        return 1 - np.sum( (occur / Y.size) ** 2 for _, occur in Counter(Y).items())

    def mean(self, X, f):
        return np.mean(np.unique(X[:, f]))
    
    def median(self, X, f):
        return np.median(np.unique(X[:, f]))
        

    # Fonction permettant de généré un ensemble de test dans le but de splitté au mieux les données
    # en deux sous_ensemble. Les labels doivent être le mieux répartie entre les deux sous_ensemble.
    # Par exemple, les NOUN d'un côté et le reste de l'autre

    def getBestTest(self, X, Y):

        nbSample, nbFeature = np.shape(X)
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
        bestYesLabel = []
        bestYesSample = []
        bestNoLabel = []
        bestNoSamples = []

        # On regarde chacune des features
        for f in range(nbFeature):

            if self.gen_test == "mean":
                test = self.mean(X, f)
            elif self.gen_test == "median":
                test = self.median(X, f)

            yes_samples = X[X[:, f] <= test]
            yes_label = Y[X[:, f] <= test]
            
            no_samples = X[X[:, f] > test]
            no_label = Y[X[:, f] > test]

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
                        bestTest = test
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
                        bestTest = test
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

                # Si on n'a pas réussi à séparé les données, on créée une feuille
                if not Yno or not Xno or not Yyes or not Xyes:
                    return Counter(Y).most_common(1)[0][0]

                # On construit la branche gauche de l'arbre avec les données ayant passé le test
                left = build_tree(Xyes, Yyes, (max_depth - 1))
                # On construit la branche droite de l'arbre avec les données n'ayant pas passé le test
                right = build_tree(Xno, Yno, (max_depth - 1))

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
        
        occurLab = Counter(Y)        
        id = list(self.classLabel.values())[-1]
        for k, _ in occurLab.most_common():
            if not k in self.classLabel:
                self.classLabel[k] = id
                self.labelClass[id] = k
                id += 1
        
        nbLabel = len(self.classLabel.values())
        cm = np.zeros( (nbLabel, nbLabel) )
        
        Ypredict = self.predict(tree, X)
        score = 0
        for ygold, ypredict in zip(Y, Ypredict):
            cm[self.classLabel[ygold], self.classLabel[ypredict]] += 1
            if ygold == ypredict:
                score += 1
        return ((score / len(Y)) * 100, cm)
        

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
            feature = list(features().keys())[feature]
            print(feature, " <= ", test, " ?")
            self.showTree(tree[1], s='Y', depth=depth+1)
            self.showTree(tree[2], s='N', depth=depth+1)


