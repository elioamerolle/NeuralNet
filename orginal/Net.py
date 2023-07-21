
from Percept import Perceptron

class NeuralNetwork(list):

    def __init__(self, input, args):
        self.input = input
        self.args = args


    def netMake(self):
        for i in range(len(self.args)):
            layList = []

            for j in range(self.args[i]):
                inptLay = (len(self) == 0)
                
                if inptLay:
                    layList.append(Perceptron(0, self, [i, j], inptLay))
                    layList[-1].setActivation(self.input[j])
                else:
                    layList.append(Perceptron(self.args[i - 1], self, [i, j]))



            self.append(layList)

    
    def getIn(self):
        return self.input

    
    def netActive(self):
        for i in range(len(self) - 1):
            print(i)
            for j in range(len(self[i + 1])):
                print(j)
                self[i + 1][j].activate()
            

    def getAct(self, depth):
        retlist = []
        
        for i in self[depth]:
            retlist.append(i.act)

        return retlist


    def printLayers(self):
        print("Depth: " + str(len(self)))

        for i in range(len(self)):
            print("Layer " + str(i) + "  has acts:")
            for j in self[i]:
                print(j.act)

            print("\n")

        


   




