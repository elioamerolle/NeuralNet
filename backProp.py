
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

        TODO:

            - Set up derivativeList for 0 < iteration
            - Set up getting the derivatives for 0 < iteration

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

    


