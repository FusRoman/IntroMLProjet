#itère sur chacun des corpus
def iterateCorpus(f, printing = True):
    for nameCorpus, corpus in da.listeCorpus.items():
        if printing:
            print("corpus : ", nameCorpus)
        for typeDS, dataset in corpus.getDataset().items():
            if printing:
                print("ensemble ", typeDS)
            f(dataset, nameCorpus, typeDS)

# met à jour les données statistiques
def updateStat(dataset, nm, tds):
    dataset.updateStat()

# affiche les statistiques basique


def printBasicStat(dataset, nm, tds):
    #baseStat.append((nm + " " + tds, len(dataset.data), dataset.nbWord, dataset.nbUniqueWords))
    print("   nombre de phrase : ", len(dataset.data))
    print("   nombre de mot : ", dataset.nbWord)
    print("   nombre de mot unique : ", dataset.nbUniqueWords)


    

# affiche les 10 mots / labels les plus fréquents de chaque corpus
def printMostFrequent(dataset, nm, tds):
    print()
    print("10 mots les plus fréquents : ")
    print(dataset.mostFrequentWord)
    print()
    print()
    print("labels les plus fréquents : ")
    print(dataset.mostFrequentLabel)
    print()
    
# affiche les 10 mots les plus ambigu de chaque corpus
def printMostAmbiguousWord(dataset, nm, tds):
    ambiguousDict, ambiguousWord = dataset.ambiguousWord()
    
    def sortSecond(val):
        return len(val[1])
    
    ambiguousWord.sort(key = sortSecond, reverse = True)
    
    print()
    print("5 mots les plus ambigu : ", ambiguousWord[:5])
    print()
    
iterateCorpus(updateStat, False)