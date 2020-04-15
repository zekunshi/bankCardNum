import random

import numpy as np
import matplotlib.pyplot as plt

# x = np.array([1,2,3,4,5,6,7,8])
# y = np.array([3,5,7,6,2,6,10,15])

f1 = open(r'.\crnn\data\accuracyRate.txt', 'r+')
allLine=f1.readlines()
allTrainRound=list()
allAccuracy=list()

# allTrainRound.append(0)
# allAccuracy.append(0)

for oneLine in allLine:
    a=oneLine.index(',')

    trainRound=int(oneLine[0:a])
    allTrainRound.append(trainRound)

    accuracy=float(oneLine[a+1:len(oneLine)-1])
    allAccuracy.append(accuracy)




plt.plot(allTrainRound,allAccuracy,'r',lw=2)