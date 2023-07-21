
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
                
                #list of lists for derivatives of perceptrons
                layerlist = []

                #iterate through perceptrons in final layer
                for i in range(len(neuralNetwork[-1])):
                    #list of Derivatives for a particular perceptron
                    perceptronDerivs = []
                    
                    #add the partial for b or dCbyda * dabydz
                    perceptronDerivs.append(derivativeList[i])
                    
                    sum = 0

                    #iterate through the connections to this particular perceptron
                    for j in range(len(neuralNetwork[-1][i].weight)):
                        #Append dCbyda * dabydz * dzbydw to the perceptronDerivs
                        perceptronDerivs.insert(0, derivativeList[i] * neuralNetwork[-2][j].getActivation())

                        #WRONG ERROR
                        sum += neuralNetwork[-1][i].weight[j]

                    #AT THIS POINT ALL DERIVATIVES FOR returnList ARE SET UP NOW ALL ABOUT MAKING USEFUL ONES GOOD
                    #WRONG
                    derivativeList[i] *= sum

                    #Update the values in helpful derivatives list

                    #add these derivatives this particular perceptron to our layerlist
                    layerlist.append(perceptronDerivs)


                returnList.append(layerlist)

                

            else:

                pass
            
            return BackPropagation.getDerivatives(derivativeList, iteration + 1, neuralNetwork, returnList)


        else:

            return returnList

    


