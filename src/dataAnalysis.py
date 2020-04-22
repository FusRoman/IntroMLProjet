import json
from os import walk
import pandas as pds
import math

# Contient une phrase et les étiquettes
# Attributs :
# - sentence :  la phrase (suite de mots)
# - labels :    les étiquettes (liste de même longueur que sentence)


class Data:
    def __init__(self, sentence, labels):
        self.sentence = sentence
        self.labels = labels

    # Itère sur les 3-grams (non uniques) de la phrase
    def iterate3Grams(self):
        for i in range(2, len(self.sentence)):
            yield (self.sentence[i - 2], self.sentence[i - 1], self.sentence[i])


# Classe qui contient les données d'un corpus et quelques méta-données précalculées
# Attributs :
# - name :                  le nom du corpus
# - data :                  liste de Data
# - nb3Grams :              nombre de 3-grams non uniques du corpus
# - nbDistinct3Grams :      nombre de 3-grams uniques du corpus
# - distinct3Grams :        ensemble des 3-grams uniques
# - nbUniqueWords :         nombre de mots uniques du corpus
# - uniqueWords :           ensemble des mots uniques du corpus
class Corpus:
    def __init__(self, name, data):
        self.name = name
        self.data = [Data(d[0], d[1]) for d in data]

        self.update3Grams()
        self.updateDistinct3Grams()
        self.updateUniqueWords()

    # Itère sur les 3-grams (non uniques) du corpus, phrase par phrase
    def iterate3Grams(self):
        for d in self.data:
            for n in d.iterate3Grams():
                yield n

    def update3Grams(self):
        self.nb3Grams = 0
        for _ in self.iterate3Grams():
            self.nb3Grams += 1

    def updateDistinct3Grams(self):
        self.nbDistinct3Grams = 0
        self.distinct3Grams = set()
        for n in self.iterate3Grams():
            if not n in self.distinct3Grams:
                self.distinct3Grams.add(n)
                self.nbDistinct3Grams += 1

    def updateUniqueWords(self):
        self.nbUniqueWords = 0
        self.uniqueWords = set()
        for d in self.data:
            for word in d.sentence:
                if not word in self.uniqueWords:
                    self.uniqueWords.add(word)
                    self.nbUniqueWords += 1


listeCorpus = []


def init():
    global listeCorpus

    def decodeData(file):
        with open(file, 'r') as fp:
            loaded_json = json.load(fp)
        return json.JSONDecoder().decode(json.dumps(loaded_json))

    corpusFolder = "../data/nlp_project_corpus/corpus/fr/"
    corpus = []
    listeCorpus = []

    for (_, _, files) in walk(corpusFolder):
        files.remove("fr.pud.dev.json")
        corpus.extend(files)

    print("Initialization...")
    for file in corpus:
        print("\tPreparing corpus " + file + "...")
        decodedData = decodeData(corpusFolder + file)
        listeCorpus.append(Corpus(file, decodedData))
        print("\tDone")
    print("Done!")


def getCorpusNames():
    return [c.name for c in listeCorpus]


def dataAnalysis():
    exampleSentence = []
    nbrSentence = []
    nbrSentenceElement = []
    nbrUniqueElement = []
    for corpus in listeCorpus:
        data = corpus.data
        occur = {}
        firstSentence = []
        # recupere les 3 premiere phrases de chaque corpus
        for sentence_label in data[0:3]:
            firstSentence.append(" ".join(sentence_label.sentence))
        exampleSentence.append(firstSentence)
        # calcul le nombre d'element (mot, ponctuation, autre...)
        # de chaque corpus
        for sentence_label in data:
            for word in sentence_label.sentence:
                occur.setdefault(word, 0)
                occur[word] += 1
        nbrSentence.append(len(data))
        nbrSentenceElement.append(sum(occur.values()))
        nbrUniqueElement.append(corpus.nbUniqueWords)
    return exampleSentence, nbrSentence, nbrSentenceElement, nbrUniqueElement

# renvoie les informations du calcul du pourcentage d'out-of-vocabulary words


def computeOOV():
    oovPercent = []
    trainFst = []
    testSnd = []

    for train in listeCorpus:
        for test in listeCorpus:
            if train != test and test.name not in trainFst:
                trainFst.append(train.name)
                testSnd.append(test.name)
                oov = test.uniqueWords - train.uniqueWords
                oovPercent.append(
                    (len(oov) / (train.nbUniqueWords + test.nbUniqueWords) * 100))
    return trainFst, testSnd, oovPercent


def count3grams(train, test):
    count = dict()
    for grams in train.iterate3Grams():
        count.setdefault(grams, 0)
        count[grams] += 1
    for grams in test.iterate3Grams():
        count.setdefault(grams, 0)
        count[grams] += 1
    return count


def computeKLDivergence(train, test):
    occur3gram = count3grams(train, test)
    dkl = 0

    # N : Nombre de 3-grams dans le corpus train et test
    N = train.nb3Grams + test.nb3Grams
    # V : Nombre de 3-grams distincts dans le corpus train et test
    V = train.nbDistinct3Grams + test.nbDistinct3Grams
    
    #calcul le denominateur de la probabilité d'apparition du 3-grams dans les ensembles de 
    # test et d'apprentissage -> #N + #V * (#d - 2) où d - 2 correspond au nombre de 3-grams
    # dans le corpus correspondant
    denomTrain = N + V * (train.nb3Grams)
    denomTest = N + V * (test.nb3Grams)

    for _, occurgrams in occur3gram.items():
        #print("grams : ", grams, " -> occur : ", occurgrams)
        p_test = (occurgrams + 1) / denomTest
        p_train = (occurgrams + 1) / denomTrain
        #print("p_test : ", p_test)
        #print("p_train : ", p_train)
        log = math.log(p_test / p_train)
        #print("log : ", log)
        dkl += p_test * log
        #print("dkl : ", dkl)
    return dkl


def computeAllKLDivergence():

    DKL = []
    trainFst = []
    testSnd = []

    for train in listeCorpus:
        for test in listeCorpus:
            if train != test:
                trainFst.append(train.name)
                testSnd.append(test.name)
                DKL.append(computeKLDivergence(train, test))

    return trainFst, testSnd, DKL
