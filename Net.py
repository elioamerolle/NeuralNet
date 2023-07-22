
from Funcs import Funcs
from backProp import BackPropagation as BP
from Percept import Perceptron

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
                    
                    layeredList.append(Perceptron(0, self, [i, j], isInputLayer))
                    # adds a perceptron
                    
                    layeredList[-1].setActivation(self.inputValues[j])
                    # sets the initial activation for the input of the layered list
                    
                else:
                    # checks if the current layer is not an input
                    
                    layeredList.append(Perceptron(self.layerDimensions[i - 1], self, [i, j]))
                    # adds a perceptron that isn't part of the input layer
            
            self.append(layeredList)
            # adds the layered list to the network
    
    def getInputValues(self):
        # returns the initial input values
        
        return self.inputValues
    
    def activate(self):
        # activates all the perceptrons in the network
        
        for i in range(len(self) - 1):
            # loops through each layers
            # print(i)
            for j in range(len(self[i + 1])):
                # loops through each perceptron
                # print(j)
                self[i + 1][j].activate()
                # activates the individual perceptron
    
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
            
            print(len(self[layerIndex]))
            print(len(returnWeightsList))
            
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


    def print(self):
        # a function for printing 
        
        print("Depth: " + str(len(self)))

        for i in range(len(self)):
            # loops through each layers
            
            print("Layer " + str(i) + "  has acts:")
            
            for j in self[i]:
                # loops through each perceptron
                
                print(j.getActivation())
                # prints the activation value
            
            print("\n")


    def update(self, drivs):
        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j].subWeight(drivs[i][j][:-1])
                self[i][j].subBias(drivs[i][j][-1])


    #function responsible for making the NN learn
    def learn(self, minibatch):
        #Takes 3D array as input of form [[[Data for input layer], [Ground Truth]],...]
        
        #List with all the derivatives
        fnlDervtvLst = []

        #iterating throuhg minibatch
        for i in minibatch:
            #Has all the derivatives also 3D of form [[[dC/dw1],...],...]
            fnlDervtvLst = Funcs.addMats(fnlDervtvLst, BP.getDerivatives([], 0, self, [], groundTruth = i[1]))

        self.update(fnlDervtvLst)

    def testLearn(self, groundTruth):
        pass




                
            
