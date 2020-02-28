import json
from os import walk
import pandas as pds

corpusFolder = "../data/nlp_project_corpus/corpus/fr/"
listeCorpus = []

for (_, _, files) in walk(corpusFolder):
    files.remove("fr.pud.dev.json")
    listeCorpus.extend(files)

def dataAnalysis():
    exampleSentence = []
    nbrSentence = []
    nbrSentenceElement = []
    for files in listeCorpus:
        occur = {}
        if(files != "fr.pud.dev.json"):
            with open(corpusFolder + files, 'r') as fp:
                loaded_json = json.load(fp)
                data = json.JSONDecoder().decode(json.dumps(loaded_json))
                exampleSentence.append(" ".join(data[0][0]))
                for sentence_label in data:
                    for word in sentence_label[0]:
                        occur.setdefault(word, 0)
                        occur[word] += 1
                nbrSentence.append(len(data))
                nbrSentenceElement.append(sum(occur.values()))
    return exampleSentence, nbrSentence, nbrSentenceElement

def decodeData(file):
    with open(file, 'r') as fp:
        loaded_json = json.load(fp)
    return json.JSONDecoder().decode(json.dumps(loaded_json))
    
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
