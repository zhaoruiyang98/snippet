import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from doctest import testmod
from queue import PriorityQueue, Queue

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
    >>> a = NPuzzle('1230'); print(a.getNeighborStates())
    ['1032', '1203']

    >>> a = NPuzzle('123456780'); print(a.getNeighborStates())
    ['123450786', '123456708']

    >>> a = NPuzzle('123045678'); print(a.getNeighborStates())
    ['023145678', '123645078', '123405678']

    >>> a = NPuzzle('123405678'); print(a.getNeighborStates())
    ['103425678', '123475608', '123045678', '123450678']

    """

    def __init__(self, state, rank=None, wordWidth=None):
        if rank is not None:
            self.rank = rank
        else:
            charList = state.split()
            total = len(charList)
            rank = int(np.sqrt(total))
            self.rank = rank

        if wordWidth is not None:
            self.wordWidth = wordWidth
            self.state = state
        else:
            charList = state.split()
            wordWidth = -1
            for x in charList:
                if len(x) > wordWidth:
                    wordWidth = len(x)
            charList = [x.center(wordWidth) for x in charList]
            self.wordWidth = wordWidth
            self.state = ' '.join(charList)

    def getRank(self):
        return self.rank

    def getState(self):
        return self.state

    def getArray(self):
        return np.fromstring(self.state, dtype='int64', sep=' ')

    def printState(self):
        """
        >>> a = NPuzzle('1 2 3 4 5 6 7 8 -'); a.printState()
        *************
        *   *   *   *
        * 1 * 2 * 3 * 
        *   *   *   *
        *************
        *   *   *   *
        * 4 * 5 * 6 * 
        *   *   *   *
        *************
        *   *   *   *
        * 7 * 8 * - * 
        *   *   *   *
        *************

        """
        rank = self.rank
        state = self.state
        wordWidth = self.wordWidth
        charList = state.split()
        charList = [x.center(wordWidth) for x in charList]
        lineWidth = rank * wordWidth + 3 * rank + 1
        print('*'*lineWidth)
        for i in range(rank):
            print(('* '+' '*wordWidth+' ')*rank+'*')
            print('* ', end='')
            for j in range(rank):
                print('%s * ' % (charList[i*rank+j]), end='')
            print('')
            print(('* '+' '*wordWidth+' ')*rank+'*')
            print('*'*lineWidth)

    def getNeighborStates(self, state=None, rank=None, wordWidth=None):
        if state is None:
            state = self.state
        if rank is None:
            rank = self.rank
        if wordWidth is None:
            wordWidth = self.wordWidth

        state = ' '+state
        minusIndex = state.index('-')
        shift = '-'.center(wordWidth).index('-')
        minusIndex = minusIndex - shift - 1
        neighborStates = []
        blockWidth = wordWidth + 1
        lineWidth = blockWidth * rank
        upIndex = minusIndex - lineWidth
        downIndex = minusIndex + lineWidth
        leftIndex = minusIndex - blockWidth
        rightIndex = minusIndex + blockWidth
        if upIndex >= 0:
            neighborStates.append((state[:upIndex] + state[minusIndex:minusIndex+blockWidth] +
                                   state[upIndex+blockWidth:minusIndex] + state[upIndex:upIndex+blockWidth] + state[minusIndex+blockWidth:])[1:])
        if downIndex < lineWidth * rank:
            neighborStates.append((state[:minusIndex] + state[downIndex:downIndex+blockWidth] +
                                   state[minusIndex+blockWidth:downIndex] + state[minusIndex:minusIndex+blockWidth] + state[downIndex+blockWidth:])[1:])
        if minusIndex % lineWidth != 0:
            neighborStates.append((state[:leftIndex] + state[minusIndex:minusIndex+blockWidth] +
                                   state[leftIndex+blockWidth:minusIndex] + state[leftIndex:leftIndex+blockWidth] + state[minusIndex+blockWidth:])[1:])
        if rightIndex % lineWidth != 0:
            neighborStates.append((state[:minusIndex] + state[rightIndex:rightIndex+blockWidth] +
                                   state[minusIndex+blockWidth:rightIndex] + state[minusIndex:minusIndex+blockWidth] + state[rightIndex+blockWidth:])[1:])
            # neighborStates[-1] = neighborStates[-1][1:]

        return neighborStates

    def getNeighbors(self):
        result = []
        for x in self.getNeighborStates():
            result.append(NPuzzle(x, rank=self.rank, wordWidth=self.wordWidth))
        return result

    def printNeighbors(self):
        l = self.getNeighbors()
        for x in l:
            x.printState()
            print('')

    def printChains(self):
        chains = self.path['chains']
        for x in chains:
            x.printState()
            print('')

    @staticmethod
    def serialize(state):
        return np.array_str(state)

    @staticmethod
    def unserialize(sState):
        return np.fromstring(sState[1:-1], dtype='int64', sep=' ')

    def solve(self, goalState, method='A*', heuristic='ManHattan'):
        # a node is represented by its state
        start = self.state

        frontier = PriorityQueue()
        frontier.put(start, 0)
        cameFrom = dict()
        costSoFar = dict()
        cameFrom[start] = None
        costSoFar[start] = 0

        while not frontier.empty():
            current = frontier.get()

            if current is goalState:
                break

            for next in self.getNeighborStates(state=current, rank=self.rank, wordWidth=self.wordWidth):
                newCost = costSoFar[current] + 1
                if (next not in costSoFar) or (newCost < costSoFar[next]):
                    costSoFar[next] = newCost
                    priority = newCost + self.heuristicCost(next, goalState, method=heuristic)
                    frontier.put(next, priority)
                    cameFrom[next] = current

        if goalState not in cameFrom:
            raise Exception(
                'Initial state cannot be transformed into goalState')

        chains = []
        node = goalState
        while node is not None:
            chains.append(NPuzzle(node, rank=self.rank,
                                  wordWidth=self.wordWidth))
            node = cameFrom[node]
        chains.reverse()
        self.path = {'goalState': goalState,
                     'chains': chains, 'steps': len(chains)}

    @staticmethod
    def getManHattanDistance(state1, state2, rank):
        list1 = state1.split()
        list2 = state2.split()
        result = 0
        for (i,x),y in zip(enumerate(list1), list2):
            if (x is '-') or (y is '-') or (x is y):
                continue
            else:
                for j,z in enumerate(list2):
                    if z is '-':
                        continue
                    if x is z:
                        x_dis = abs(j-i)%rank
                        y_dis = abs(j//rank -i//rank)
                        result += x_dis + y_dis
                        break
        return result

        

    @staticmethod
    def getHammingDistance(state1, state2):
        list1 = state1.split()
        list2 = state2.split()
        result = 0
        for x,y in zip(list1, list2):
            if (x is not '-') and (y is not '-') and (x is not y):
                result += 1
        return result

    def heuristicCost(self, state1, state2, method):
        if method is 'Hamming':
            return self.getHammingDistance(state1, state2)
        elif method is "ManHattan":
            return self.getManHattanDistance(state1, state2, self.rank)
        else:
            return 0


if __name__ == '__main__':
    testmod()
