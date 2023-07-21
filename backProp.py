
from Funcs import Funcs


class BackPropagation(object):
    
    def getFirstDerivatives(neuralNetwork, grounTruth):
        retList = []
        for i in range(neuralNetwork.layerDimensions[-1]):
            perceptron  = neuralNetwork[-1][i]
            retList.append(2*(perceptron.getActivation() - grounTruth[i])*Funcs.sigDeriv(perceptron.preSigmoidActivation()))

        return retList

    

    """
    How we would call function: getDerivatives([dC/da,..., dC/da], 0, self, []])

    def func
        if iter < max
            - Check for first run:
                - derivativeList = generate the initial deriv list
                - and figure out all the for dC/dw (there is not sum because each weight can only 
                influence the cost via a single neuron) 

            - iterate through neurons
                - Find sum of dC/da*...*dz/da + ... (iterate through the layer ahead)

                - set sum equal to dC/db

                iterate through the weights of current neuron
                    - returnList values =  derivativeList Values multiplied by dz/dw
                
            - iterate throguh derivativeList and multiply elemnets by da/dz (increasing the number of elements for each comp)

            func(iter + 1, storedList, derivativeList)

        

    """

    def getDerivatives(derivativeList, iteration, neuralNetwork, returnList, firstCall = False, groundTruth = None):
        if iteration < len(neuralNetwork):
            if firstCall:
                derivativeList = BackPropagation.getFirstDerivatives(neuralNetwork, groundTruth)
                

            else:
                pass

            BackPropagation.getDerivatives(derivativeList, iteration + 1, neuralNetwork, returnList)

        else:
            return returnList

    


