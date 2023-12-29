
from Funcs import Funcs


class BackPropagation(object):
    
    """
   
    Goal:
        -   need correct chain of derivatives to properly alter weights and biases
            this would include acounting for softmax also get rid of of sigmoid in last layer

    Procedure:
        -   Currently have softmax working for activation but backprop
        -   only need to change getFirstDerivatives

        Cost is not a function of the softax layer so we need dC/ds*ds/da*da/dz
    
        ds/da = d/da(e^a/(e^a + const)) = e^a * (e^a + const)^-1 + (-((e^a + const)^-2) * e^2a)

        so we should really be dealing with the z (presigact) instead of a (sigact)

        dC/ds * ds/da * da/dw

        2*(neuralNetwork.softMax[i] - groundTruth[i]) * d(neuralNetwork.softMax[i])/da

    """

    #returns the derivates dCbyda * dabydz when we are starting out 
    def getFirstDerivatives(neuralNetwork, groundTruth):
        retList = []
        
        for i in range(neuralNetwork.layerDimensions[-1]):

            sum = 0
            for j in range(neuralNetwork.layerDimensions[-1]):
                #dCbyds = 2 * (neuralNetwork.softMax[j] - groundTruth[j])
                
                dCbyds = (1/10)*(neuralNetwork.softMax[j] - groundTruth[j])/(neuralNetwork.softMax[j] * (1 - neuralNetwork.softMax[j]))


                if i == j:
                    dsbydz = neuralNetwork.softMax[j] * (1 - neuralNetwork.softMax[j])
                else:
                    dsbydz = - neuralNetwork.softMax[i] * neuralNetwork.softMax[j]
                
                sum += dCbyds*dsbydz

            
            retList.append(sum)

        return retList

    

    """
    How we would call function: getDerivatives([dC/da,..., dC/da], 0, self, []])

    def func
        if iter < max
            - Check for first run:
                - derivativeList = generate the initial deriv list
                - and figure out all the for dC/dw (there is not sum because each weight can only 
                influence the cost via a single neuron) 

            else
                - iterate through neurons
                    - Find sum of dC/da*...*dz/da + ... (iterate through the layer ahead)

                    - set sum equal to dC/db

                    iterate through the weights of current neuron
                        - returnList values =  derivativeList Values multiplied by dz/dw
                    
            - iterate throguh derivativeList and multiply elemnets by da/dz (increasing the number of elements for each comp)

            func(iter + 1, storedList, derivativeList)

    """

    def getDerivatives(derivativeList, iteration, neuralNetwork, returnList, groundTruth = None):
        if iteration < len(neuralNetwork) - 1:
            if iteration == 0:
                #get the initial set of of helpful derivatives
                derivativeList = BackPropagation.getFirstDerivatives(neuralNetwork, groundTruth)
            
            #due to how python indexes lists with negative numbers
            currentLayer = - (iteration + 1)
            
            #list of lists for derivatives of perceptrons
            layerlist = []

            #iterate through perceptrons in final layer
            for i in range(len(neuralNetwork[currentLayer])):
                #list of Derivatives for a particular perceptron
                perceptronDerivs = []
                
                #iterate through the connections to this particular perceptron
                for j in range(len(neuralNetwork[currentLayer][i].weight)):
                    #Append dCbyda * dabydz * dzbydw to the perceptronDerivs
                    perceptronDerivs.append(derivativeList[i] * neuralNetwork[currentLayer - 1][j].getActivation())

                #add the partial for b or dCbyda * dabydz
                perceptronDerivs.append(derivativeList[i])

                layerlist.append(perceptronDerivs)

            #AT THIS POINT ALL DERIVATIVES FOR returnList ARE SET UP NOW ALL ABOUT MAKING USEFUL ONES GOOD
            
            #The length of our new helpful list the length of the next colum
            length = len(neuralNetwork[currentLayer - 1])

            newderivativeList = [0] * length

            returnList.insert(0, layerlist)


            if iteration < len(neuralNetwork) - 2:
                
                for e in range(length):
                    for m in range(len(derivativeList)):
                        newderivativeList[e] += derivativeList[m] * neuralNetwork[currentLayer][m].weight[e]
                    
                    newderivativeList[e] *= Funcs.sigDeriv(neuralNetwork[currentLayer - 1][e].preSigmoidActivation())


            
            return BackPropagation.getDerivatives(newderivativeList, iteration + 1, neuralNetwork, returnList)


        else:

            return returnList


    def printDerivList(derivs):
        for i in range(len(derivs)):
            print("LAYER: " + str(i + 1) + "\n")
            for j in range(len(derivs[i])):
                print("perceptron in layer: " + str(i + 1) + " and level: " + str(j) + " activations: \n")
                
                print("weights: \n")
                print(derivs[i][j][:-1])

                print("\n")


                print("bias: \n")
                print(derivs[i][j][-1])

    


