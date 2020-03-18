import json
from os import walk
import pandas as pds

corpusFolder = "../data/nlp_project_corpus/corpus/fr/"
listeCorpus = []
decodedData = []

for (_, _, files) in walk(corpusFolder):
    files.remove("fr.pud.dev.json")
    listeCorpus.extend(files)


def decodeData(file):
    with open(file, 'r') as fp:
        loaded_json = json.load(fp)
    return json.JSONDecoder().decode(json.dumps(loaded_json))


for file in listeCorpus:
    decodedData.append(decodeData(corpusFolder + file))


def dataAnalysis():
    exampleSentence = []
    nbrSentence = []
    nbrSentenceElement = []
    nbrUniqueElement = []
    for data in decodedData:
        occur = {}
        uniqueElement = set()
        firstSentence = []
        # recupere les 3 premiere phrases de chaque corpus
        for sentence_label in data[0:3]:
            firstSentence.append(" ".join(sentence_label[0]))
        exampleSentence.append(firstSentence)
        # calcul le nombre d'element (mot, ponctuation, autre...)
        # de chaque corpus
        for sentence_label in data:
            for word in sentence_label[0]:
                uniqueElement.add(word)
                occur.setdefault(word, 0)
                occur[word] += 1
        nbrSentence.append(len(data))
        nbrSentenceElement.append(sum(occur.values()))
        nbrUniqueElement.append(len(uniqueElement))
    return exampleSentence, nbrSentence, nbrSentenceElement, nbrUniqueElement

def uniqueWord(data):
    uniqueWord = set()
    for sentence_label in data:
        for word in sentence_label[0]:
            uniqueWord.add(word)
    return uniqueWord

# renvoie les informations du calcul du pourcentage d'out-of-vocabulary words
def computeOOV():
    oovPercent = []
    trainFst = []
    testSnd = []
    uniqWordAllCorpus = []
    for data in decodedData:
        uniqWordAllCorpus.append(uniqueWord(data))
    
    for train, nameTrainCorpus in zip(uniqWordAllCorpus, listeCorpus):
        lenUniqtrain = len(train)
        for test, nameTestCorpus in zip(uniqWordAllCorpus, listeCorpus):
            if nameTrainCorpus != nameTestCorpus and nameTestCorpus not in trainFst:
                trainFst.append(nameTrainCorpus)
                testSnd.append(nameTestCorpus)
                oov = train - test
                oovPercent.append( (len(oov) / (lenUniqtrain + len(test))) * 100 )
    return trainFst, testSnd, oovPercent

"""
def computeKLdivergence():
    for train, trainCorpus in zip(decodedData, listeCorpus):
        for test, testCorpus in zip(decodedData, listeCorpus):
            if nameTrainCorpus != nameTestCorpus and nameTestCorpus not in trainFst:
                


computeKLdivergence()
"""


"""
decode_ftb_train = decodeData(fr_ftb_train)
decode_ftb_test = decodeData(fr_ftb_test)
decode_ftb_dev = decodeData(fr_ftb_dev)

analysis_train = analysisData(decode_ftb_train)
analysis_test = analysisData(decode_ftb_test)
analysis_dev = analysisData(decode_ftb_dev)

print("Nombre de phrase :")
print(" Trainset =>", analysis_train[0])
print(" Testset =>", analysis_test[0])
print(" Devset =>", analysis_dev[0])
print("Nombre de mot :")
print(" Trainset =>", analysis_train[2])
print(" Testset =>", analysis_test[2])
print(" Devset =>", analysis_dev[2])

train_word = set(analysis_train[1].keys())
print(analysis_train[1].keys())
test_word = set(analysis_train[1].keys())
dev_word = set(analysis_dev[1].keys())

oov_test = test_word - train_word
oov_dev = dev_word - train_word

print("Out of vocabulary word (OOV) :")
print(" OOV Testset =>", oov_test)
#print(" OOV Devset =>", oov_dev)
"""
