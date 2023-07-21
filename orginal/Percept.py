
from Funcs import Funcs

class Perceptron:

    def __init__(self, nW, Net, loc, input = False):
        #Known variables 
        self.nW = nW
        self.input = input
        self.Net = Net
        self.loc = loc

        #Unkown or are a function of known
        self.b = Funcs.rando() * 5
        self.weight = Funcs.randList(nW)
        self.activation = None


    def setWB(self, w, b):
        if not self.input:
            self.w = w
            self.b = b

        else:
            raise Exception("Error: cannot set weights and biasis to input (has no meaning)")


    def setActivation(self, act):
        if self.input:
            self.activation = act

        else:
            raise Exception("Error: cannot set activation for non input nodes")


    @property
    def act(self):
        return self.activation


    def activate(self):
        depth = self.loc[0]

        print(self.weight)
        print(self.Net.getAct(depth - 1))
        
        z = Funcs.dotPr(self.weight, self.Net.getAct(depth - 1)) + self.b

        self.activation = Funcs.sig(z)






            
