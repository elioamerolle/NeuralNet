import matplotlib.pyplot as plt

import random
from tkinter import DoubleVar
from tokenize import Double
import numpy as np 
import math
from termcolor import colored




class Funcs:
    

    def exponential_decay(epoch, initial_lr, decay_rate):
        return initial_lr * math.exp(-decay_rate * epoch)

    def flatten(l1):
        retlist = []

        for i in l1:
            for j in i:
                retlist.append(j)

        return retlist


    def rando():
        return np.random.uniform(-math.sqrt(6/(74)), math.sqrt(6/(74)))


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


    def shuffle(lst):
    
        arr = np.array(lst, dtype=object)

        np.random.shuffle(arr)

        return list(arr)


    def sPerRound(decimal):
        return str(round(decimal * 100, 2)) + "%"

    def graphing(dataMSE, dataLogLoss):
        # from ChatGPT 3.5

        # Plotting cost function calues after each epoch 
        # Create a range of x values based on the length of the lists
        x_values = range(len(dataMSE))

        plt.ion()

        # Plotting the data
        plt.plot(x_values, dataMSE, label='MSE', color='blue')  # Blue line for dataMSE
        plt.plot(x_values, dataLogLoss, label='Log Loss', color='orange')  # Orange line for dataLogLoss

        # Adding labels and title
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title('Overlay of dataMSE and dataLogLoss')

        # Adding a legend
        plt.legend()

        # Show the plot
        plt.show()

        # from ChatGPT 3.5 end

    def full_learn_cycle(myNet, mnistClean, nImages, miniSize, count, learnR, decay_rate, dataMSE, dataLogLoss):
        dataClean = []
        # populate data clean note that its alreadu in minibatches
        Funcs.load_up_data(dataClean, mnistClean, nImages, miniSize)

        # updates the learning rate
        learnR = Funcs.exponential_decay(count, learnR, decay_rate)

        # Learning step
        for j in range(len(dataClean)):
            myNet.learn(dataClean[j], learnR, dataMSE, dataLogLoss)
        

    def load_up_data(dataClean, mnistClean, nImages, miniSize):
        # Cuts up MNIST into batches and puts into dataClean
        for i in range(nImages//miniSize):
            miniBatch = []
        
            for j in range(miniSize):
                    miniBatch.append(mnistClean[i*miniSize + j])

            dataClean.append(miniBatch)


    def find_success(myNet, mnistRaw, start_ind, end_ind):
        countWrong = 0
        success = [0] * 10

        nTestIm = end_ind - start_ind

        for i in range(nTestIm):

            target = mnistRaw.target[i + start_ind]

            myNet.activate(Funcs.flatten(mnistRaw.images[i + start_ind]))

            maxInd = 0

            for e in range(len(myNet.getActivation(-1))):
                    if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[e]:
                        maxInd = e
                        

            if maxInd != target:
                    countWrong += 1

            else:
                    success[target] += 1

        return (nTestIm - countWrong)/nTestIm


    def asker(myNet, mnist):
        while True:

            plt.close("all")

            index = int(input("WHAT INDEX FROM MNIST WOULD YOU LIKE TO SAMPLE ON OUR NET \n"))

            targetVal = mnist.target[index]

            print("THE TARGET VALUE FOR THIS INDEX IS " + str(targetVal) + "\n")
            
            plt.ion()

            plt.matshow(mnist.images[index])

            myNet.activate(Funcs.flatten(mnist.images[index]), False)


            maxInd = 0
            for i in range(len(myNet.getActivation(-1))):
                    if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[i]:
                        maxInd = i
            
            print("Confidences")
            for i in range(len(myNet.softMax)):
                prctg = round(myNet.softMax[i] * 100 , 2)

                if maxInd == i:
                    if targetVal == maxInd:
                        print(colored(str(i) + " : " + str(prctg) + "%", 'green'))
                    else:
                        print(colored(str(i) + " : " + str(prctg) + "%", 'red'))

                else:
                    print(str(i) + " : " + str(prctg) + "%")

            print("THE NEURAL NET HAS GUESSED " + str(maxInd) + "\n")
            
            plt.show()


            print("\n\n\n")

            inputStr = input("WOULD YOU LIKE TO CONTINUE (y/n) \n")

            if inputStr == "n":
                exit()
            

    def mnistExpectedBin(x):
        retlist = [0] * 10

        for i in range(10):
            if i == x:
                retlist[i] = 1

        return retlist


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





