import json
from Net import NeuralNetwork
from backProp import BackPropagation as BP
from Funcs import Funcs
from Vars import Vars


l1 = [0.2, 0.5, 0.6]

myNet = NeuralNetwork(l1, [4])



print("SETTING PERCEPTRON DATA \n")

# parses json data

myNet.create()


print("BEFORE ACTIVATED \n")

myNet.print()


print("ACTIVATE INDICES \n")

myNet.activate()

print("AFTER ACTIVATED \n")

myNet.print()


#print(BP.getFirstDerivatives(myNet, [1, 0.5, 1, 0.5]))

print("Here is the deriv list coming \n")
print("NN length " + str(len(myNet)))
print(BP.getDerivatives([], 0, myNet, [], [1, 0.5, 1, 0.5]))


"""

l3 = []

l4 = [[[1,652,1],[24,4,6],[43,31,34]],[[1,4,6],[1,4,6],[24,3,6]],[[1,4,6],[24,5,7],[3,5,6]]]


print(Funcs.addMats(l3, l4))



Future plans:
- Backpropagation

"""
