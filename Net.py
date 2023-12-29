
from Funcs import Funcs
from backProp import BackPropagation as BP
from Percept import Perceptron
import math
import copy


"""
TODO: 
 - Input values seems unecisary??
 - fix the learn functions 

"""

class NeuralNetwork(list):
    # the class for the Neural Network that inherits properties and methods from the list data type
    
    def __init__(self, inputValues, layerDimensions):
        # the initalizer for the input values and the layer dimensions
        
        self.inputValues = inputValues
        # stores the input values
        
        self.layerDimensions = layerDimensions
        # stores the layered dimension
        
        self.layerDimensions.insert(0, len(inputValues))
        # adds the length of the input values

        self.softMax = [0] * layerDimensions[-1]
    
    def create(self):
        # a function to fill the neural network
        
        if len(self) != 0:
            # if it's already defined
            
            print("NeuralNetwork.create: Warning: Neural Network has already been defined (likely by the setPerceptronData method). Function will not run.")
            
            return
            
        
        for i in range(len(self.layerDimensions)):
            # loops through all the layers
            
            layeredList = []
            # creates the layered list to return
            
            for j in range(self.layerDimensions[i]):
                # loops through each perceptron
                
                isInputLayer = (len(self) == 0)
                # checks if it's of type input layer
                
                if isInputLayer:
                    # checks if the current layer is an input
                    
                    layeredList.append(Perceptron(0, self, [i, j], True))
                    # adds a perceptron
                    
                    layeredList[-1].setActivation(self.inputValues[j])
                    # sets the initial activation for the input of the layered list
                    
                else:
                    # checks if the current layer is not an input
                    
                    isOutputLayer = (i == len(self.layerDimensions) - 1)

                    layeredList.append(Perceptron(self.layerDimensions[i - 1], self, [i, j], False, isOutputLayer))
                    # adds a perceptron that isn't part of the input layer
            
            self.append(layeredList)
            # adds the layered list to the network
    
    def getInputValues(self):
        # returns the initial input values
        
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
            print("we start activated")

        for i in range(len(self[0])):
            self[0][i].setActivation(input[i])

        for i in range(len(self) - 1):
            # loops through each layers

            for j in range(len(self[i + 1])):
                # loops through each perceptron

                self[i + 1][j].activate()
                # activates the individual perceptron
        
        self.activateSoftMax()

        if printBool:
            print("finished activated")
    
    def getPerceptronData(self):
        # gets all the total weight
        
        returnList = []
        # creates a list to return
        
        for layerIndex in range(len(self)):
            # loops through each layers
            
            returnWeightsList = []
            # creates a list to return
            
            for perceptron in self[layerIndex]:
                # loops through each perceptron
                
                perceptronData = {
                  # creates the perceptron data
                  
                  "weight": perceptron.weight,
                  # stores the weights
                  
                  "bias": perceptron.bias
                  # stores the bias
                }
                
                returnWeightsList.append(perceptronData)
                # adds the weight of the perceptron
            
            
            returnList.append(returnWeightsList)
            # adds the weights list
        
        return returnList
        # returns the list
    
    def setPerceptronData(self, data):
        # sets all the total weight
        
        self.create()
        # creates the data
        
        for i in range(len(self)):
            # loops through each layers
            
            for j in range(len(self[i])):
                # loops through each perceptron
                
                perceptronData = data[i][j]
                # gets the perceptron data
                
                if i != 0:
                  # if it's not in the input values
                  
                  self[i][j].setWeightBias(perceptronData["weight"], perceptronData["bias"])
                  # sets the weight and bias
        
    
    def getLayer(self, depth):
        # a function to get the layer
        
        return self[depth]
    
    
    def getPerceptron(self, layerIndex, perceptronIndex):
        # a function to get a perceptron
        
        return self[layerIndex][perceptronIndex]
    
    def getActivation(self, depth):
        # gets the values for the lists
        
        returnList = []
        # creates a list to return
        
        for i in self[depth]:
            # loops through each perceptron
            
            returnList.append(i.getActivation())
            # adds the activation value to the return list
        
        return returnList


    def print(self, beforeSig = False):
        # a function for printing 
        print("Depth: " + str(len(self)))

        for i in range(len(self)):
            # loops through each layers
            
            if beforeSig:
                print("Layer " + str(i) + "  has before Sig acts:")
            else:
                print("Layer " + str(i) + "  has acts:")

            for j in range(len(self[i])):
                # loops through each perceptron
                
                if beforeSig:
                    if i != 0:
                        print(self[i][j].preSigmoidActivation())
                else:
                    print(self[i][j].getActivation())

                # prints the activation value
            
            print("\n")


    def update(self, drivs, step):
        for i in range(len(self) - 1):
            for j in range(len(self[i + 1])):
            
                try:
                    self[i + 1][j].subWeight(drivs[i][j][:-1], step)
                    self[i + 1][j].subBias(drivs[i][j][-1], step)
                
                except IndexError:
                    print("Error index i " + str(i))
                    #print(len(drivs))
                    print(drivs)

                    print("Error index j " + str(j))
                    raise(IndexError("Wrong index"))

    #function responsible for making the NN learn
    def learn(self, minibatch, groundTruths, data):
        #Takes 3D array as input of form [[[Data for input layer], [Ground Truth]],...]
        
        #List with all the derivatives
        fnlDervtvLst = []

        data.append(self.getMiniBatchCost(minibatch, groundTruths))

        #iterating through minibatch
        for i in range(len(minibatch)):
            #Has all the derivatives also 3D of form [[[dC/dw1],...],...]
            self.activate(minibatch[i])

            fnlDervtvLst = Funcs.addMats(fnlDervtvLst, BP.getDerivatives([], 0, self, [], groundTruth = groundTruths[i]))


        step = 0.0002

        trashNet = copy.deepcopy(self)

        #print("RIGHT BEFORE PROPB UPDATE")
        #print(fnlDervtvLst)

        trashNet.update(fnlDervtvLst, step)
        
        sumTrash = 0
        sum = 0
        
        #Probab
        for i in range(len(minibatch)):
            trashNet.activate(minibatch[i])
            sumTrash += Funcs.cost(trashNet.getActivation(-1), groundTruths[i]) 
            self.activate(minibatch[i])
            sum += Funcs.cost(self.getActivation(-1), groundTruths[i]) 
            

        if sumTrash < sum:
            #print("should get better")
            
            self.update(fnlDervtvLst, step)

        else:
            #print("might not get better")

            #step *= 0.1

            self.update(fnlDervtvLst, step)
    
        

    def getCost(self, groundTruth):
        sum = 0

        #print("GT: " + str(groundTruth))
        #print("Out: " + str(self.getActivation(-1)))


        for i in range(len(self[-1])):
            if type(groundTruth) == list:
                sum += (groundTruth[i] - self[-1][i].getActivation()) ** 2
            else:
                #print("GT: " + str(groundTruth))
                #print("Activation: " + str(self[-1][i].getActivation()))

                sum += (groundTruth - self[-1][i].getActivation()) ** 2

        
        return sum

    
    def getCostFromIn(self, input, groundTruth):
        sum = 0

        self.activate(input)

        for i in range(len(self[-1])):
            if type(groundTruth) == list:
                sum += (groundTruth[i] - self[-1][i].getActivation()) ** 2
            else:
                #print("GT: " + str(groundTruth))
                #print("Activation: " + str(self[-1][i].getActivation()))

                sum += (groundTruth - self[-1][i].getActivation()) ** 2

        
        return sum

        


    def getMiniBatchCost(self, minibatch, groundTruths):
        sum = 0
        for i in range(len(minibatch)):
            sum += self.getCostFromIn(minibatch[i], groundTruths[i])
        
        return sum


    def testLearn(self, input, derivs, groundTruth):
        print("Initial cost: " + str(self.getCost(groundTruth)))
        
        for i in range(1000):
            step = 1

            trashNet = copy.deepcopy(self)

            trashNet.update(derivs, step)
            
            trashNet.activate(input)
            
            print("trash" + str(trashNet.getCost(groundTruth)))
            print("real" + str(self.getCost(groundTruth)))

            nextBreak = False

            if trashNet.getCost(groundTruth) < self.getCost(groundTruth):
                self = trashNet
            else:
                while trashNet.getCost(groundTruth) > self.getCost(groundTruth):
                    print("in while ")

                    step *= 0.1

                    trashNet.update(derivs, step)
            
                    trashNet.activate()
                    
                    if step < 0.00001:
                        nextBreak = True
                        break

            if nextBreak:
                break

            print("Cost after update " + str(i) + ": " + str(self.getCost(groundTruth)))

    

        




                
            
