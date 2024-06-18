
from Funcs import Funcs
from backProp import BackPropagation as BP
from Percept import Perceptron
import math
import copy



# the class for the Neural Network that inherits properties and methods from the list data type
class NeuralNetwork(list):
    
    # the initalizer for the input values and the layer dimensions
    def __init__(self, layerDimensions):
        
        # stores the layered dimension
        self.layerDimensions = layerDimensions
        
        # where softmax values stored
        self.softMax = [0] * layerDimensions[-1]

        self.succPrctg = 0


    # a function to fill the neural network
    def create(self):

        # if it's already defined
        if len(self) != 0:
            
            print("NeuralNetwork.create: Warning: Neural Network has already been defined (likely by the setPerceptronData method). Function will not run.")
            
            return
            
        
        for i in range(len(self.layerDimensions)):
            
            layeredList = []
            
            for j in range(self.layerDimensions[i]):
                
                isInputLayer = (len(self) == 0)
                
                if isInputLayer:
                    
                    layeredList.append(Perceptron(0, self, [i, j], True))
                
                    
                else:
                    
                    isOutputLayer = (i == len(self.layerDimensions) - 1)

                    layeredList.append(Perceptron(self.layerDimensions[i - 1], self, [i, j], False, isOutputLayer))
            
            self.append(layeredList)
    
    def getInputValues(self):
        
        return self.inputValues
    

    #Softmax Does not work it is not right setup need to use exp
    def activateSoftMax(self):
        sum = 0

        for i in self[-1]:
            sum += math.exp(i.getActivation())

        for j in range(len(self.softMax)):
            self.softMax[j] = math.exp(self[-1][j].getActivation())/sum


    def activate(self, input, printBool = False):
        # activates all the perceptrons in the network
        if printBool:
            print("activation start")

        for i in range(len(self[0])):
            self[0][i].setActivation(input[i])

        for i in range(len(self) - 1):

            for j in range(len(self[i + 1])):

                self[i + 1][j].activate()
        
        self.activateSoftMax()

        if printBool:
            print("activation end")
    
    def getPerceptronData(self):
        
        returnList = []
        
        for layerIndex in range(len(self)):
            
            returnWeightsList = []
            
            for perceptron in self[layerIndex]:
                
                perceptronData = {
                  
                  "weight": perceptron.weight,
                  
                  "bias": perceptron.bias
                }
                
                returnWeightsList.append(perceptronData)
            
            
            returnList.append(returnWeightsList)
        
        return returnList
    
    def setPerceptronData(self, data):
        
        self.create()
        
        for i in range(len(self)):
            
            for j in range(len(self[i])):
                
                perceptronData = data[i][j]
                
                if i != 0:
                  
                  self[i][j].setWeightBias(perceptronData["weight"], perceptronData["bias"])
        
    
    def getLayer(self, depth):
        
        return self[depth]
    
    
    def getPerceptron(self, layerIndex, perceptronIndex):
        
        return self[layerIndex][perceptronIndex]
    
    def getActivation(self, depth):
        
        returnList = []
        
        for i in self[depth]:
            
            returnList.append(i.getActivation())
        
        return returnList


    def print(self, beforeSig = False):

        print("Depth: " + str(len(self)))

        for i in range(len(self)):
            
            if beforeSig:
                print("Layer " + str(i) + "  has before Sig acts:")
            else:
                print("Layer " + str(i) + "  has acts:")

            for j in range(len(self[i])):
                
                if beforeSig:
                    if i != 0:
                        print(self[i][j].preSigmoidActivation())
                else:
                    print(self[i][j].getActivation())

            
            print("\n")


    def update(self, drivs, step):
        for i in range(len(self) - 1):
            for j in range(len(self[i + 1])):
            
                try:
                    self[i + 1][j].subWeight(drivs[i][j][:-1], step)
                    self[i + 1][j].subBias(drivs[i][j][-1], step)
                
                except IndexError:
                    print("Error index i " + str(i))
                    print(drivs)

                    print("Error index j " + str(j))
                    raise(IndexError("Wrong index"))


    def learn(self, minibatch, learnR, dataMSE, dataLogLoss):
        
        #List with all the derivatives
        fnlDervtvLst = []

        dataMSE.append(self.getMiniBatchCost(minibatch, "S"))
        dataLogLoss.append(self.getMiniBatchCost(minibatch, "L"))

        #iterating through minibatch
        for i in range(len(minibatch)):
            #Has all the derivatives also 3D of form [[[dC/dw1],...],...]
            self.activate(minibatch[i][0])

            fnlDervtvLst = Funcs.addMats(fnlDervtvLst, BP.getDerivatives([], 0, self, [], groundTruth = minibatch[i][1]))

        
        self.update(fnlDervtvLst, learnR)


    def getCostLog(self, input, groundTruth):
        sum = 0

        self.activate(input)

        for i in range(len(self[-1])):
            if type(groundTruth) == list:
                sum += groundTruth[i] * math.log(self.softMax[i]) + (1 - groundTruth[i]) * math.log(1 - self.softMax[i])

            else:
                sum += groundTruth * math.log(self.softMax[i]) + (1 - groundTruth[i]) * math.log(1 - self.softMax[i])

        return -(1/len(self.softMax)) * sum

    
    def getCostL2(self, input, groundTruth):
        sum = 0

        self.activate(input)

        for i in range(len(self[-1])):
            if type(groundTruth) == list:
                sum += (groundTruth[i] - self.softMax[i]) ** 2

            else:
                sum += (groundTruth - self.softMax[i]) ** 2

        return sum

    

    def getMiniBatchCost(self, minibatch, strIn):
        sum = 0
        for i in range(len(minibatch)):
            if strIn == "S":
                sum += self.getCostL2(minibatch[i][0], minibatch[i][1])

            else:
                sum += self.getCostLog(minibatch[i][0], minibatch[i][1])


        return sum

