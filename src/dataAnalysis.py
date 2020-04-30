import json
from os import walk
import pandas as pds
import math
from collections import Counter


def mergeAddDict(dict1, dict2):
    # Fusionne deux dictionnaires en ajoutant les valeurs des clé commune dans la liste
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = value + dict1[key]

    return dict3


# Contient une phrase et les étiquettes
# Attributs :
# - sentence :  la phrase (suite de mots)
# - labels :    les étiquettes (liste de même longueur que sentence)
# - dataSet :   Référence vers le dataset contenant cette phrase.
class Data:
    def __init__(self, sentence, labels, dataSet):
        self.sentence = sentence
        self.labels = labels
        self.dataSet = dataSet

    # Itère sur les 3-grams (non uniques) de la phrase
    def iterate3Grams(self):
        for i in range(2, len(self.sentence)):
            yield (self.sentence[i - 2], self.sentence[i - 1], self.sentence[i])

    # Itère sur les 2-grams (non unique) de la phrase
    def iterate2grams(self):
        for i in range(1, len(self.sentence)):
            yield (self.sentence[i - 1], self.sentence[i])


# Classe qui contient les données d'un corpus et quelques méta-données précalculées
# Attributs :
# - name :                  le nom du corpus
# - data :                  liste de Data
# - nb3Grams :              nombre de 3-grams non uniques du corpus
# - nbDistinct3Grams :      nombre de 3-grams uniques du corpus
# - distinct3Grams :        ensemble des 3-grams uniques
# - nbWord :                nombre de mots du corpus
# - nbUniqueWords :         nombre de mots uniques du corpus
# - uniqueWords :           ensemble des mots uniques du corpus
# - occurLabel :            ensemble des labels unique du corpus et leurs occurences
# - occurTrigram :          nombre d'occurence des trigrams sous la forme d'un dictionnaire
# - occurBigram :           nombre d'occurence des bigrams sous la forme d'un dictionnaire
# - languageModel :         calcule les probabilité correspondant au modele du langage
class DataSet:
    def __init__(self, name, data):
        self.name = name
        self.data = [Data(d[0], d[1], self) for d in data]

    def updateStat(self):
        self.updateUniqueWords()
        self.computeOccur()
        self.computeLanguageModel()
        self.updateNbLabel()

    # Itère sur les 3-grams (non uniques) du corpus, phrase par phrase
    def iterate3Grams(self):
        for d in self.data:
            for n in d.iterate3Grams():
                yield n

    # Itère sur les 2-grams (non uniques) du corpus, phrase par phrase
    def iterate2grams(self):
        for d in self.data:
            for n in d.iterate2grams():
                yield n

    def wordsByLabel(self):
        wordLabel = dict()
        for d in self.data:
            for w, l in zip(d.sentence, d.labels):
                wordLabel.setdefault(l, set())
                wordLabel[l].add(w)
        return wordLabel

    def updateNbLabel(self):
        self.occurLabel = Counter()
        for d in self.data:
            self.occurLabel.update(d.labels)
        self.mostFrequentLabel = self.occurLabel.most_common(5)

    def ambiguousWord(self):
        ambiguousWord = dict()
        for d in self.data:
            for w, l in zip(d.sentence, d.labels):
                ambiguousWord.setdefault(w, list())
                ambiguousWord[w].append(l)
        ambiguousWord = {k : Counter(v) for k, v in ambiguousWord.items()}
        mostAmbiguousWord = [(k, v)
                             for k, v in ambiguousWord.items() if len(v) > 1]
        return ambiguousWord, mostAmbiguousWord

    def updateUniqueWords(self):
        self.occurWord = Counter()
        for d in self.data:
            self.occurWord.update(d.sentence)
        self.nbUniqueWords = len(self.occurWord.keys())
        self.uniqueWords = set(self.occurWord.keys())
        self.nbWord = sum(self.occurWord.values())
        self.mostFrequentWord = self.occurWord.most_common(5)

    def computeOccur(self):
        self.occurTrigram = Counter(self.iterate3Grams())
        self.occurBigram = Counter(self.iterate2grams())
        self.nbDistinct3Grams = len(self.occurTrigram.keys())
        self.distinct3Grams = set(self.occurTrigram.keys())
        self.nb3Grams = sum(self.occurTrigram.values())

    # Calcul le modele trigram de ce dataset.
    # P(w_i | w_i-2, w_i-1) = count({w_i-2, w_i-1, w_i}) / count({w_i-2, w_i-1})
    def computeLanguageModel(self):
        self.languageModel = dict()
        for trigram, occurTrigram in self.occurTrigram.items():
            occurBigram = self.occurBigram[(trigram[0], trigram[1])]
            self.languageModel[trigram] = occurTrigram / occurBigram

    # Calcul la perplexité du dataset suivant la formule :
    # PP(ds) = exp( -sum_{i = 1}^{N}(P(w_i | w_i-2, w_i-1)) / N )
    def computePerplexity(self):
        res = 0
        for _, proba in self.languageModel.items():
            res -= math.log(proba)
        return math.exp(res / self.nbWord)


class Corpus:
    def __init__(self, nameCorpus, DataSet):
        self.nameCorpus = nameCorpus
        self.trainExist = False
        self.testExist = False
        self.devExist = False
        for type, ds in DataSet:
            if type == "train":
                self.trainExist = True
                self.trainDataSet = ds
            if type == "test":
                self.testExist = True
                self.testDataSet = ds
            if type == "dev":
                self.devExist = True
                self.devDataSet = ds

    def getDataset(self):
        res = dict()
        if self.trainExist:
            res["train"] = self.trainDataSet
        if self.testExist:
            res["test"] = self.testDataSet
        if self.devExist:
            res["dev"] = self.devDataSet
        return res

    # Calcul le pourcentage d'out of vocabulary word entre l'ensemble de test et l'ensemble
    # d'apprentissage de ce corpus

    def computeCorpusOOV(self):
        if self.trainExist:
            if self.testExist:
                oovWithTest = self.testDataSet.uniqueWords - self.trainDataSet.uniqueWords
                percentOOVWithTest = len(
                    oovWithTest) / (self.trainDataSet.nbUniqueWords + self.testDataSet.nbUniqueWords) * 100
                if self.devExist:
                    oovWithDev = self.devDataSet.uniqueWords - self.trainDataSet.uniqueWords
                    percentOOVWithDev = len(
                        oovWithDev) / (self.trainDataSet.nbUniqueWords + self.devDataSet.nbUniqueWords) * 100
                    return percentOOVWithTest, percentOOVWithDev
                return percentOOVWithTest, None

    # Calcul le pourcentage d'oov entre l'ensemble d'apprentissage de ce corpus et un autre DataSet
    # otherDataSet doit être une instance de DataSet
    def computeOOV(self, otherDataSet):
        if self.trainExist:
            oov = otherDataSet.uniqueWords - self.trainDataSet.uniqueWords
            percentOOV = len(
                oov) / (self.trainDataSet.nbUniqueWords + otherDataSet.nbUniqueWords) * 100
            return percentOOV

    # Calcul la divergence de KullBack-Leibler entre l'ensemble de test et d'apprentissage
    # de ce corpus
    def computeCorpusKLDivergence(self):
        if self.trainExist and self.testExist:
            dkl = 0
            allDistinctTrigram = self.trainDataSet.distinct3Grams.union(
                self.testDataSet.distinct3Grams)

            # N : Nombre de 3-grams dans le corpus train et test
            N = self.trainDataSet.nb3Grams + self.testDataSet.nb3Grams
            # V : Nombre de 3-grams distincts dans le corpus train et test
            V = self.trainDataSet.nbDistinct3Grams + self.testDataSet.nbDistinct3Grams

            # calcul le denominateur de la probabilité d'apparition du 3-grams dans les ensembles de
            # test et d'apprentissage -> #N + #V * (#d - 2) où d - 2 correspond au nombre de 3-grams
            # dans le corpus correspondant
            denomTrain = N + V * (self.trainDataSet.nb3Grams)
            denomTest = N + V * (self.testDataSet.nb3Grams)

            for trigram in allDistinctTrigram:
                # Calcul de l'additive smoothing pour l'ensemble de test et d'apprentissage
                p_test = (self.testDataSet.occurTrigram.get(
                    trigram, 0) + 1) / denomTest
                p_train = (self.trainDataSet.occurTrigram.get(
                    trigram, 0) + 1) / denomTrain
                log = math.log2(p_test / p_train)
                dkl += p_test * log
            return dkl

    # Calcul la divergence de KullBack-Leibler entre l'ensemble d'apprentissage de ce corpus
    # et un autre dataSet
    # otherDataSet doit être une instance de DataSet

    def computeKLDivergence(self, otherDataSet):
        if self.trainExist:
            dkl = 0
            allDistinctTrigram = self.trainDataSet.distinct3Grams.union(
                otherDataSet.distinct3Grams)

            # N : Nombre de 3-grams dans le corpus train et test
            N = self.trainDataSet.nb3Grams + otherDataSet.nb3Grams
            # V : Nombre de 3-grams distincts dans le corpus train et test
            V = self.trainDataSet.nbDistinct3Grams + otherDataSet.nbDistinct3Grams

            # calcul le denominateur de la probabilité d'apparition du 3-grams dans les ensembles de
            # test et d'apprentissage -> #N + #V * (#d - 2) où d - 2 correspond au nombre de 3-grams
            # dans le corpus correspondant
            denomTrain = N + V * (self.trainDataSet.nb3Grams)
            denomTest = N + V * (otherDataSet.nb3Grams)

            for trigram in allDistinctTrigram:
                # Calcul de l'additive smoothing pour l'ensemble de test et d'apprentissage
                p_test = (otherDataSet.occurTrigram.get(
                    trigram, 0) + 1) / denomTest
                p_train = (self.trainDataSet.occurTrigram.get(
                    trigram, 0) + 1) / denomTrain
                log = math.log2(p_test / p_train)
                dkl += p_test * log
            return dkl

    def computePerplexityCorpus(self):
        if self.trainExist:
            if self.testExist:
                if self.devExist:
                    return self.trainDataSet.computePerplexity(), self.testDataSet.computePerplexity(), self.devDataSet.computePerplexity()
                else:
                    return self.trainDataSet.computePerplexity(), self.testDataSet.computePerplexity()
            else:
                return self.trainDataSet.computePerplexity()
        else:
            if self.testExist:
                if self.devExist:
                    return self.testDataSet.computePerplexity(), self.devDataSet.computePerplexity()
                else:
                    return self.testDataSet.computePerplexity()


listeCorpus = {}


def init():
    global listeCorpus

    def decodeData(file):
        with open(file, 'r') as fp:
            loaded_json = json.load(fp)
        return json.JSONDecoder().decode(json.dumps(loaded_json))

    corpusFolder = "../data/nlp_project_corpus/corpus/fr/"
    corpus = []
    listeCorpus = {}

    corpusType = dict()

    for (_, _, files) in walk(corpusFolder):
        files.remove("fr.pud.dev.json")
        corpus.extend(files)

    print("Initialization...")
    for file in corpus:
        print("\tPreparing corpus " + file + "...")
        l = file.split(".")
        nameDataSet = l[1]
        typedataSet = l[2]
        decodedData = decodeData(corpusFolder + file)
        ds = DataSet(file, decodedData)
        corpusType.setdefault(nameDataSet, [])
        corpusType[nameDataSet].append((typedataSet, ds))
        print("\tDone")

    for nameCorpus, ds in corpusType.items():
        listeCorpus[nameCorpus] = Corpus(nameCorpus, ds)
    print("Done!")
