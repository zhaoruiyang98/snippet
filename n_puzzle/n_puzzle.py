import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from doctest import testmod

# class Cell:
#     def __init__(self, index, n):
#         bottom, left = 0.1, 0.1
#         width, height = 0.8, 0.8
#         top, right = bottom + height, left + width
#         vstep = (top-bottom)/n
#         hstep = (right-left)/n
#         xIndex = index % n
#         yIndex = index - index % n
#         self.bottomleft = np.array([])


class NPuzzle:
    """
    >>> a = NPuzzle('321')
    Traceback (most recent call last):
        ...
    ValueError: The length of input state should be a perfect square

    >>> a = NPuzzle(123)
    Traceback (most recent call last):
        ...
    TypeError: The type of state should be str

    >>> a = NPuzzle('4321')
    Traceback (most recent call last):
        ...
    ValueError: Illegal state: each element should be in the range of 0 to 3

    >>> a = NPuzzle('1230')

    >>> a = NPuzzle('1102')
    Traceback (most recent call last):
        ...
    ValueError: Illegal state: each element should appear only once

    >>> a = NPuzzle('1230'); print(a.getNeighborStates())
    ['1032', '1203']

    >>> a = NPuzzle('123456780'); print(a.getNeighborStates())
    ['123450786', '123456708']

    >>> a = NPuzzle('123045678'); print(a.getNeighborStates())
    ['023145678', '123645078', '123405678']

    >>> a = NPuzzle('123405678'); print(a.getNeighborStates())
    ['103425678', '123475608', '123045678', '123450678']

    """

    def __init__(self, state):
        self._checkIndex(state)

    def _checkIndex(self, state):
        if type(state) is not str:
            raise TypeError("The type of state should be str")
        rank = int(np.sqrt(len(state)))
        if len(state) != rank**2:
            raise ValueError(
                "The length of input state should be a perfect square")
        for i in range(len(state)):
            if str(i) not in state:
                raise ValueError(
                    "Illegal state: each element should be in the range of 0 to %d" % (len(state)-1))
            if state.count(str(i)) != 1:
                raise ValueError(
                    "Illegal state: each element should appear only once")
        self.rank = rank
        self.state = state

    def getRank(self):
        return self.rank

    def getState(self):
        return self.state

    def getArray(self):
        result =  np.array(list(map(int,list(self.state))))
        result = result.reshape((self.rank,self.rank))
        return result

    def printState(self):
        rank = self.rank
        linewidth = 4*rank + 1
        print('*'*linewidth)
        for i in range(rank):
            print('*   '*rank+'*')
            print('* ', end='')
            for j in range(rank):
                print('%d * ' % (int(self.state[i*rank+j])), end='')
            print('')
            print('*   '*rank+'*')
            print('*'*linewidth)
    
    def arrayToState(self, array):
        rank = self.rank
        state = array.reshape((rank**2))
        state = list(state)
        state = list(map(str,state))
        return ''.join(state)

    def getNeighborStates(self):
        stateArray = self.getArray()
        rank = self.rank
        zeroIndex = np.unravel_index(np.argmin(stateArray), stateArray.shape)

        neighborIndexes = []
        for i,j in [(-1,0),(1,0),(0,-1),(0,1)]:
            newi = zeroIndex[0]+i
            newj = zeroIndex[1]+j
            if (0 <= newi < rank) and (0 <= newj < rank):
                neighborIndexes.append((newi,newj))
        
        neighborStates = []
        for x in neighborIndexes:
            tempArray = stateArray.copy()
            temp = tempArray[x]
            tempArray[x] = tempArray[zeroIndex]
            tempArray[zeroIndex] = temp
            neighborStates.append(self.arrayToState(tempArray))

        return neighborStates

    def getNeighbors(self):
        result = []
        for x in self.getNeighborStates():
            result.append(NPuzzle(x))
        return result

    def printNeighbors(self):
        l = self.getNeighbors()
        for x in l:
            x.printState()
            print('')



if __name__ == '__main__':
    testmod()
    # a = NPuzzle('462301578')
    # a.printNeighbors()