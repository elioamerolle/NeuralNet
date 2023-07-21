import random
import numpy as np 


class Funcs:

    def rando():
        return (random.random() - 0.5) * 2


    def sig(x):
        return 1/(1 + np.exp(-x))


    def dotPr(l1, l2):
    
        arr1 = np.array(l1)
        arr2 = np.array(l2)

        return np.dot(arr1, arr2)


    def randList(n):
        retList = []

        for i in range(n):
            retList.append(Funcs.rando())

        return retList


