
from Funcs import Funcs

class Perceptron:

    def __init__(self, numberOfWeight, neuralNetwork, indices, isInputLayer = False, isOutputLayer = False):
        
        # defined properties:
        self.numberOfWeight = numberOfWeight
        self.isInputLayer = isInputLayer
        self.isOutputLayer = isOutputLayer

        self.neuralNetwork = neuralNetwork
        self.indices = indices

        # randomized data:
        self.bias = 0
        self.weight = Funcs.randList(numberOfWeight)
        self.activation = None
        
        if isInputLayer:
            # if there is an input layer
            
            self.bias = 0
            # sets the bias to 0

    def setWeight(self, weight):
        # a function to set the weight
        if not self.isInputLayer:
            if len(weight) == len(self.weight):
                self.weight = weight
            
            else:
                raise Exception("Error: inputted list is not the same size as the weight list")

        else:
            raise Exception("Error: cannot set weight for input nodes")


    def setBias(self, bias):
        # a function to set the bias
        if not self.isInputLayer:
            self.bias = bias

        else:
            raise Exception("Error: cannot set bias for input nodes")


    def subWeight(self, deltaW, step):
        
        if len(deltaW) == len(self.weight):
            for i in range(len(deltaW)):
                self.weight[i] -= deltaW[i] * step

        else:
            raise Exception("Error: inputted list is not the same size as the weight list")


    def subBias(self, deltaB, step):
        self.bias -= deltaB * step


    def setWeightBias(self, weight, bias):
        # a function to set both the weight bias
        if not self.isInputLayer:
            self.setWeight(weight)
            self.setBias(bias)

        else:
            raise Exception("Error: cannot set weights and biasis to input (has no meaning)")


    def setActivation(self, act):
        # sets the activation value
        
        if self.isInputLayer:
            self.activation = act

        else:
            raise Exception("Error: cannot set activation for non input nodes")

    def getActivation(self):
        # returns the activation value
        return self.activation

    def preSigmoidActivation(self):
        # gets the layer depth
        layerDepth = self.indices[0]

        # gets the dot product of the weights and the bias
        return Funcs.dotPr(self.weight, self.neuralNetwork.getActivation(layerDepth - 1)) + self.bias



    def activate(self):
        if self.isOutputLayer:
            #Because we dont want nonlinearity before softmax
            self.activation = self.preSigmoidActivation()

        else:
            # gets the sig value of the dot product
            self.activation = Funcs.sig(self.preSigmoidActivation())
            
