
from Funcs import Funcs


class BackPropagation(object):
    
    #returns the derivates dCbyda * dabydz when we are starting out
    def getFirstDerivatives(neuralNetwork, grounTruth):
        retList = []
        for i in range(neuralNetwork.layerDimensions[-1]):
            perceptron  = neuralNetwork[-1][i]

            dCbyda = 2*(perceptron.getActivation() - grounTruth[i])
            dabydz = Funcs.sigDeriv(perceptron.preSigmoidActivation())

            retList.append(dCbyda*dabydz)

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
                print("ENTERED IF")
                derivativeList = BackPropagation.getFirstDerivatives(neuralNetwork, groundTruth)
                
                #list of lists for derivatives of perceptrons
                layerlist = []

                #iterate through perceptrons in final layer
                for i in range(len(neuralNetwork[-1])):
                    #list of Derivatives for a particular perceptron
                    perceptronDerivs = []
                    
                    #iterate through the connections to this particular perceptron
                    for j in range(len(neuralNetwork[-1][i].weight)):
                        #Append dCbyda * dabydz * dzbydw to the perceptronDerivs
                        perceptronDerivs.append(derivativeList[i] * neuralNetwork[-2][j].getActivation())

                    #add the partial for b or dCbyda * dabydz
                    perceptronDerivs.append(derivativeList[i])
                    layerlist.append(perceptronDerivs)


                returnList.append(layerlist)

                print("returnList 1")
                print(returnList)

                return BackPropagation.getDerivatives(derivativeList, iteration + 1, neuralNetwork, returnList)

            else:
                pass
            
            print("returnList 2")
            print(returnList)

        else:
            print("returnList 3")
            print(returnList)

            print("\n\n\n")

            return returnList

    


