
from Net import NeuralNetwork
from Funcs import Funcs

l1 = [0.2, 0.5, 0.6]

myNet = NeuralNetwork(l1, [len(l1), 4, 2])

myNet.netMake()


print("BEFORE ACTIVATED \n")

myNet.printLayers()

myNet[2][0].setWB([0.2, 0.4, 0.1], 10)

print("ACTIVATE INDICES \n")

myNet.netActive()


print("AFTER ACTIVATED \n")

myNet.printLayers()



