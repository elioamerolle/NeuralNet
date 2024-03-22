import matplotlib.pyplot as plt

import random
from tkinter import DoubleVar
from tokenize import Double
import numpy as np 
import math



class Funcs:

    def exponential_decay(epoch, initial_lr, decay_rate):
        return initial_lr * math.exp(-decay_rate * epoch)


    def shuffle(lst):

        arr = np.array(lst)

        np.random.shuffle(arr)

        return list(arr)


    def asker(myNet, mnist):
        while True:
    
            index = int(input("WHAT INDEX FROM MNIST WOULD YOU LIKE TO SAMPLE ON OUR NET \n"))

            print("THE TARGET VALUE FOR THIS INDEX IS " + str(mnist.target[index]) + "\n")
            
            plt.ion()

            plt.matshow(mnist.images[index])

            myNet.activate(Funcs.flatten(mnist.images[index]), True)

            print("THE SOFTMAX LAYER IS AS FOLLOWS \n")

            print(myNet.softMax)

            maxInd = 0
            for i in range(len(myNet.getActivation(-1))):
                    if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[i]:
                        maxInd = i

            print("THE NEURAL NET HAS GUESSED " + str(maxInd) + "\n")
            
            plt.show()


            print("\n\n\n")


    def mnistExpectedBin(x):
        retlist = [0] * 10

        for i in range(10):
            if i == x:
                retlist[i] = 1

        return retlist



    def flatten(l1):
        retlist = []

        for i in l1:
            for j in i:
                retlist.append(j)

        return retlist


    def rando():
        return np.random.uniform(-math.sqrt(6/(74)), math.sqrt(6/(74)))


    def cost(outPut, grndTrth):
        numTypes = [int, DoubleVar, float]
        

        if (isinstance(outPut, list) and isinstance(grndTrth, list)) and (len(outPut) == len(grndTrth)):
            sum = 0
            for i in range(len(outPut)):
                sum += (outPut[i] - grndTrth[i]) ** 2

            return sum

        if type(outPut) in numTypes and type(grndTrth) in numTypes:
            return (outPut - grndTrth) ** 2

        else:
            raise TypeError("Either the types are not compatible or the grndTrth list and outPut list are not the same size")


    def sig(x):
        return 1/(1 + np.exp(-x))


    def sigDeriv(x):
        return Funcs.sig(x)*(1 - Funcs.sig(x))


    def dotPr(l1, l2):
    
        arr1 = np.array(l1)
        arr2 = np.array(l2)

        return np.dot(arr1, arr2)


    def randList(n):
        retList = []

        for i in range(n):
            retList.append(Funcs.rando())

        return retList
    

    def addMats(l1,l2):
        numTypes = [int, DoubleVar]

        if len(l1) == 0:
            return l2
                
        elif len(l1) == len(l2):
            retlist = l1

            for i in range(len(l1)):
                for j in range(len(retlist[i])):
                    for e in range(len(retlist[i][j])):
                        retlist[i][j][e] = l1[i][j][e] + l2[i][j][e]
        
            return retlist
        
        else:
            TypeError("list lengths are not the same")




