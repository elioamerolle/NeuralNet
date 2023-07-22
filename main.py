from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from TrainingData import TrainingData
import json
from Net import NeuralNetwork
from backProp import BackPropagation as BP
from Funcs import Funcs
from Vars import Vars



mnist = load_digits()

l1 = [0] * 64

myNet = NeuralNetwork(l1, [50,25,10])

myNet.create()

dataClean = []
expectedClean = []

#80 minibatches of 20
for i in range(1):
      miniBatch = []
      expVals = []
      for j in range(1):
            miniBatch.append(Funcs.flatten(mnist.images[i*20 + j]))
            expVals.append(Funcs.mnistExpectedBin(mnist.target[i*20 + j]))

      expectedClean.append(expVals)
      dataClean.append(miniBatch)


for i in range(100):
      myNet.learn(dataClean[i], expectedClean[i])






"""


                              =============================

                              GOOD CODE FOR MNIST

                              =============================


mnist = load_digits()

l1 = [0] * 64

myNet = NeuralNetwork(l1, [50,25,10])

myNet.create()

dataClean = []
expectedClean = []

#We will use 1600 images

#80 minibatches of 20
for i in range(80):
      miniBatch = []
      expVals = []
      for j in range(20):
            miniBatch.append(Funcs.flatten(mnist.images[i*20 + j]))
            expVals.append(Funcs.mnistExpectedBin(mnist.target[i*20 + j]))

      expectedClean.append(expVals)
      dataClean.append(miniBatch)


for i in range(len(dataClean)):
      myNet.learn(dataClean[i], expectedClean[i])



while True:

      index = int(input("WHAT INDEX FROM MNIST WOULD YOU LIKE TO SAMPLE ON OUR NET \n"))

      print("THE TARGET VALUE FOR THIS INDEX IS " + str(mnist.target[index]) + "\n")

      myNet.activate(Funcs.flatten(mnist.images[index]), True)

      print("THE OUTPUT LAYER IS AS FOLLOW \n")

      print(myNet.getActivation(-1))

      maxInd = 0
      for i in range(len(myNet.getActivation(-1))):
            if myNet.getActivation(-1)[maxInd] < myNet.getActivation(-1)[i]:
                  maxInd = i

      print("THE NEURAL NET HAS GUESSED " + str(maxInd) + "\n")

      continueStr = input("WOULD YOU LIKE TO TRY ANOTHER (y/n)" + "\n")

      if continueStr == "n":
            break

      print("\n\n\n\n\n")


                              =============================

                              GOOD CODE FOR MNIST

                              =============================





plt.imshow(mnist.images[0]);
plt.show()
print(mnist.target[0]);



TD = TrainingData()




myNet.create()

frstRun = True

for i in range(2**10 - 1):
      start = i * 10
      finish = start + 10

      minibatch = TD.images[start : finish]
      expected = TD.expectedOutputs[start : finish]
      
      
      testImage = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]

      testOut = 0

    

      myNet.activate(testImage)

      if frstRun:
            print("Initial activation for test image " + str(myNet.getActivation(-1)))
            frstRun = False

      print("Cost BEFORE LEARN: " + str(myNet.getCost(testOut)))

      myNet.learn(minibatch, expected)

      #print("Test Image: " + str(testImage))

      myNet.activate(testImage)

      print("Cost AFTER LEARN: " + str(myNet.getCost(testOut)))
      

print("Final activation for test image " + str(myNet.getActivation(-1)))


minibatch = TD.images[20]
expected = TD.expectedOutputs[20]

myNet.activate(minibatch)


print("Current Cost: " + str(myNet.getCost(expected)))

for i in range(10):
      
      derivs = BP.getDerivatives([], 0, myNet, [], groundTruth = expected)

      myNet.testLearn(minibatch, derivs, expected)

      myNet.activate(minibatch)


"""


"""

# parses json data



print("BEFORE ACTIVATED \n")

myNet.print()


print("ACTIVATE INDICES \n")

#myNet.activate()

print("AFTER ACTIVATED \n")

myNet.print()


#print(BP.getFirstDerivatives(myNet, [1, 0.5, 1, 0.5]))

print("Here is the deriv list coming \n")
print("NN length " + str(len(myNet)))
derivs = BP.getDerivatives([], 0, myNet, [], [1, 0.5, 1, 0.5])

#BP.printDerivList(BP.getDerivatives([], 0, myNet, [], [1, 0.5, 1, 0.5]))

myNet.testLearn(derivs, [1, 0.5, 1, 0.5])





l3 = []

l4 = [[[1,652,1],[24,4,6],[43,31,34]],[[1,4,6],[1,4,6],[24,3,6]],[[1,4,6],[24,5,7],[3,5,6]]]


print(Funcs.addMats(l3, l4))



Future plans:
- Backpropagation

"""
