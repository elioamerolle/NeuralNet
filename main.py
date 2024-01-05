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


#                       =====================    Hyperparemters     =====================

# Size of each layer (input and output are determined)
layerDimensions = [64, 30, 10]

# Number of images we will use
nImages = 1600

miniSize = 10

# Number of Epochs
nEpochs = 3

# learning rate
learnR = 0.1

# decay rate (exponential loss) I think something is off
decay_rate = 2

#                       =====================     Hyperparemters    =====================

mnistRaw = load_digits()

if inStr == "t":

      # initilization of Neural Network
      myNet = NeuralNetwork(layerDimensions)
      myNet.create()

      # Loading the data set
      mnistClean = []

      for i in range(nImages):
            mnistClean.append([Funcs.flatten(mnistRaw.images[i]), Funcs.mnistExpectedBin(mnistRaw.target[i])])


      # Saves cost function per mini batch
      dataMSE = []
      dataLogLoss = []

      # Epoch counter to time loss of learning rate
      count = 0

      for i in tqdm(range(nEpochs), desc="Learning Data Set"):
            
            dataClean = []

            # updates the learning rate
            learnR = Funcs.exponential_decay(count, learnR, decay_rate)
            
            # Cuts up MNIST into batches and puts into dataClean
            for i in range(nImages//miniSize):
                  miniBatch = []
            
                  for j in range(miniSize):
                        miniBatch.append(mnistClean[i*miniSize + j])

                  dataClean.append(miniBatch)

            # Learning step
            for j in range(len(dataClean)):
                  myNet.learn(dataClean[j], learnR, dataMSE, dataLogLoss)

            mnistClean = Funcs.shuffle(mnistClean)
            count += 1



      #                 AFTER THIS POINT ITS JUST TO CHECK THE NEURAL, NET TRAINING IS COMPLETE


      # from ChapGPT 3.5

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

      # from ChapGPT 3.5 end


      print("TESTING ON UNSEEN IMAGES")

      # Arrays to track success
      countWrong = 0
      success = [0] * 10

      nTestIm = 150

      for i in tqdm(range(nTestIm), desc="finding success rate"):

            target = mnistRaw.target[1601 + i]

            myNet.activate(Funcs.flatten(mnistRaw.images[1601 + i]))

            maxInd = 0

            for e in range(len(myNet.getActivation(-1))):
                  if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[e]:
                        maxInd = e
                        

            if maxInd != target:
                  countWrong += 1

            else:
                  success[target] += 1

      print("success arr: ")
      print(success)
      print("success rate is " + str(((nTestIm - countWrong)/nTestIm) * 100) + "%")

      myNet.succPrctg = (((nTestIm - countWrong)/nTestIm) * 100)


      print("TESTING ON SEEN IMAGES")

      # Arrays to track success
      countWrong = 0
      success = [0] * 10

      nTestIm = 150

      for i in tqdm(range(1600), desc="finding success rate"):
      
            target = mnistRaw.target[i]

            myNet.activate(Funcs.flatten(mnistRaw.images[i]))

            maxInd = 0

            for e in range(len(myNet.getActivation(-1))):
                  if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[e]:
                        maxInd = e
                        

            if maxInd != target:
                  countWrong += 1

            else:
                  success[target] += 1

      print("success arr: ")
      print(success)
      print("success rate is " + str(((1600 - countWrong)/1600) * 100) + "%")


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


