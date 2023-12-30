from random import shuffle
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from TrainingData import TrainingData
import json
from Net import NeuralNetwork
from backProp import BackPropagation as BP
from Funcs import Funcs
from Vars import Vars

from tqdm import tqdm


#                       =====================    Hyperparemters     =====================

# Size of each layer (input and output are determined)
layerDimensions  = [64, 30, 10]

# Number of images we will use
nImages = 1600

miniSize = 20

# Number of Epochs
nEpochs = 20

# learning rate
learnR = 0.1

# decay rate (exponential loss)
decay_rate = 2

#                       =====================     Hyperparemters    =====================

# initilization of Neural Network
myNet = NeuralNetwork(layerDimensions)
myNet.create()

# Loading the data set
mnistRaw = load_digits()
mnistClean = []

for i in range(nImages):
      mnistClean.append([Funcs.flatten(mnistRaw.images[i]), Funcs.mnistExpectedBin(mnistRaw.target[i])])


# Saves cost function per mini batch
data = []

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
            myNet.learn(dataClean[j], learnR, data)

      mnistClean = Funcs.shuffle(mnistClean)
      count += 1



#                 AFTER THIS POINT ITS JUST TO CHECK THE NEURAL, NET TRAINING IS COMPLETE


# Plotting cost function calues after each epoch
plt.ion()
plt.plot(data)

# Arrays to track success
messUp = []
success = [0] * 10

for i in tqdm(range(1600), desc="finding success rate"):

      target = mnistRaw.target[i]

      myNet.activate(Funcs.flatten(mnistRaw.images[i]))

      maxInd = 0

      for e in range(len(myNet.getActivation(-1))):
            if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[e]:
                  maxInd = e
                  

      if maxInd != target:
            messUp.append(i)

      else:
            success[target] += 1

print("success arr: ")
print(success)
print("success rate is " + str(((1600 - len(messUp))/1600) * 100) + "%")
#print("bad Indexes: " + str(messUp))


Funcs.asker(myNet, mnistRaw)


