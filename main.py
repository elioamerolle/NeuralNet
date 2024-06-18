from random import shuffle
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from TrainingData import TrainingData
import json
from Net import NeuralNetwork
from backProp import BackPropagation as BP
from Funcs import Funcs
from Vars import Vars

import pickle
import os


from tqdm import tqdm

inStr = input("Would you like to test saved version or or train a new one s/t \n")

#                       =====================    Personal Preferences     =====================

extra_prints = False

#                       =====================    Personal Preferences     =====================


#                       =====================    Hyperparemters     =====================

# Size of each layer (input and output are determined)
layerDimensions = [64, 15, 10]

# Number of images we will use
nImages = 1600

miniSize = 20

# Number of Epochs
nEpochs = 3

# learning rate
learnR = 0.1

# decay rate 
decay_rate = 1

#If we repeat
tryMulti = 3

#                       =====================     Hyperparemters    =====================

mnistRaw = load_digits()

if inStr == "t":

      # Loading the data set
      mnistClean = []

      for i in range(nImages):
            mnistClean.append([Funcs.flatten(mnistRaw.images[i]), Funcs.mnistExpectedBin(mnistRaw.target[i])])


      # Saves cost function per mini batch
      dataMSE = []
      dataLogLoss = []

      successOld = 0

      # We make several neural nets to pick the best one
      for i in tqdm(range(tryMulti), desc="Neural Nets"):
            

            # initilization of Neural Network
            myNet = NeuralNetwork(layerDimensions)
            myNet.create()

            # Epoch counter to time loss of learning rate
            count = 0

            for i in tqdm(range(nEpochs), desc="Learning Data Set"):
                  Funcs.full_learn_cycle(myNet, mnistClean, nImages, miniSize, count, learnR, decay_rate, dataMSE, dataLogLoss)
                  
                  count += 1



            #                 AFTER THIS POINT ITS JUST TO CHECK THE NEURAL, NET TRAINING IS COMPLETE

            if(tryMulti == 1):
                  #If we are only testing one neural net I want to see the loss over time
                  Funcs.graphing(dataMSE, dataLogLoss)

            else:
                  # Arrays to track success
                  current_success = Funcs.find_success(myNet, mnistRaw, 1601, 1751)

                  if extra_prints:
                        print("success on Unseen "+ str(current_success))

                                          #TEST WITHOUT PICKING THE BEST ONE 
                  
                  if(successOld < current_success):
                        BestNet = myNet
                        successOld = current_success
            

      myNet = BestNet

      # Arrays to track success
      print("success on seen " + str(Funcs.find_success(myNet, mnistRaw, 0, 1600)))
      print("success on unseen " + str(Funcs.find_success(myNet, mnistRaw, 1601, 1751)))


      with open('net.pkl','rb') as netPikl:
            oldNet = pickle.load(netPikl)

      try:
            if oldNet.succPrctg < myNet.succPrctg:
                  inStr2 = input("This Neural Net is the best version you have found would you like to save it y/n \n")
                  
                  if inStr2 == "y":
                        os.remove("net.pkl")
                        with open('net.pkl','wb') as netPikl:
                              pickle.dump(myNet, netPikl)
      except:
            inStr2 = input("Would you like to save it y/n \n")
            
            if inStr2 == "y":
                  os.remove("net.pkl")
                  with open('net.pkl','wb') as netPikl:
                        pickle.dump(myNet, netPikl)


      Funcs.asker(myNet, mnistRaw)

else:

      with open('net.pkl','rb') as netPikl:
            myNet = pickle.load(netPikl)
      
      print("The success rate is: " + str(myNet.succPrctg) + "%")
      print("indecese 1600 < are unseen")

      Funcs.asker(myNet, mnistRaw)


