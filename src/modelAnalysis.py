import time
import numpy as np
import math
import matplotlib.pyplot as plt
import dataAnalysis as da
import DecisionTreeClassifier as dtree
import itertools


def analyzeDecisionTree(nameDataSet, maximum_depth=30, verbose=False):

    trainSet, testSet = da.listeCorpus[nameDataSet].trainDataSet, da.listeCorpus[nameDataSet].testDataSet

    Xtrain, Ytrain = dtree.buildFeature(trainSet)
    Xtest, Ytest = dtree.buildFeature(testSet)

    decisionTree = dtree.DecisionTreeClassifier(Xtrain, Ytrain)

    bestScore = -1
    bestTree = None
    bestParam = None
    bestConfusionMatrix = None

    fig, axs = plt.subplots(2, 2, figsize=(25, 14))
    coordAxesX = 0
    coordAxesY = 0

    fig.suptitle("Accuracy analysis of decision tree on " +
                 nameDataSet + " dataset")

    pastTotTime = time.time()

    for split_criterion in ["gini", "entropy"]:
        for gen_test in ["median", "mean"]:
            
            if verbose:
                print("split_criterion : ", split_criterion,
                      " / gen_test : ", gen_test)
                
                
            actualBestScore = -1
            listScoreTrain = []
            listScoreTest = []
            
            for i in range(1, maximum_depth + 1):

                decisionTree.setNewParam((i, split_criterion, gen_test))

                # fit tree
                past = time.time()
                tree = decisionTree.fit()
                if verbose:
                    print("      training time : ", time.time() - past)

                # compute score
                scoreTest, cmTest = decisionTree.modelScore(tree, Xtest, Ytest)
                scoreTrain, _ = decisionTree.modelScore(
                    tree, Xtrain, Ytrain)

                listScoreTrain.append(scoreTrain)
                listScoreTest.append(scoreTest)

                # get best score
                if scoreTest > bestScore:
                    bestScore = scoreTest
                    bestTree = tree
                    bestConfusionMatrix = cmTest
                    bestParam = (i, split_criterion, gen_test)
                    
                    
                if verbose and scoreTest > actualBestScore:
                    print("      Max depth : ", i, "score on train : ", scoreTrain,
                              "  Actual score on test : ", scoreTest, " Highscore !!")
                elif verbose:
                    print("      Max depth : ", i, "score on train : ",
                          scoreTrain, "  Actual score on test : ", scoreTest)
                    
                    
            # make the axis
            axs[coordAxesX, coordAxesY].plot(
                range(1, maximum_depth + 1), listScoreTrain, label="trainSet")
            axs[coordAxesX, coordAxesY].plot(
                range(1, maximum_depth + 1), listScoreTest, label="testSet ")
            axs[coordAxesX, coordAxesY].set_title(
                split_criterion + " / " + gen_test)

            axs[coordAxesX, coordAxesY].annotate(("bestScore : " + str(bestScore)),
                                                 (bestParam[0], bestScore),
                                                 xycoords='data',
                                                 textcoords='axes fraction',
                                                 xytext=(0.95, 0.18),
                                                 arrowprops=dict(facecolor='black',
                                                                 shrink=0.04, width=1, headwidth=8),
                                                 horizontalalignment='right', verticalalignment='top')

            axs[coordAxesX, coordAxesY].legend()
            coordAxesY += 1

        coordAxesX += 1
        coordAxesY = 0

    for ax in axs.flat:
        ax.set(xlabel='max_depth', ylabel='accuracy')
        ax.label_outer()

    cm_fig = showConfusionMatrix(bestConfusionMatrix, decisionTree.classLabel.keys()
                                 , title="DecisionTree confusion matrix with param ("
                                 + "max_depth:" + str(bestParam[0])
                                 + ", split_criterion:" + str(bestParam[1])
                                 + ", gen_test:" + str(bestParam[2])
                                 + ")")

    fig.show()

    cm_fig.show()

    fig.savefig("decisionTreeAnalysis_on_" + nameDataSet)
    cm_fig.savefig("decisionTree_confusionMatrix_on_" + nameDataSet)

    print("decisionTreeAnalysis on " + nameDataSet + " finished in " +
          str(time.time() - pastTotTime) + " secondes")

    print("best score is ", bestScore, " with param ", bestParam)

    return bestTree, bestParam


def showConfusionMatrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(25, 14))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if math.isnan(cm[i, j]):
                plt.text(j, i, "X",
                         horizontalalignment="center")
            else:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))

    return fig
