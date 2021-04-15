from math import sqrt
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


def is_equal(state1, state2):
    return state1 == state2

def get_neighbors(state):
    rank = int(sqrt(len(state)))
    index = state.index(0)
    neighbors = []

    up_index = index - rank
    down_index = index + rank
    left_index = index - 1
    right_index = index + 1

    if up_index >= 0:
        temp = state.copy()
        temp[index], temp[up_index] = temp[up_index], temp[index]
        neighbors.append(temp)
    if down_index < rank * rank:
        temp = state.copy()
        temp[index], temp[down_index] = temp[down_index], temp[index]
        neighbors.append(temp)
    if index % rank != 0:
        temp = state.copy()
        temp[index], temp[left_index] = temp[left_index], temp[index]
        neighbors.append(temp)
    if right_index % rank != 0:
        temp = state.copy()
        temp[index], temp[right_index] = temp[right_index], temp[index]
        neighbors.append(temp)
    
    return neighbors

def get_manhattan_distance(state, goal, rank):
    result = 0
    for (i, x), _ in zip(enumerate(state), goal):
        if (x == _) or (x == 0):
            continue
        for j, y in enumerate(goal):
            if x == y:
                x_dis = abs(j-i)%rank
                y_dis = abs(j//rank -i//rank)
                result += x_dis + y_dis
                break
    return int(result)

def get_conflict_manhattan_distance(state, goal, rank):
    raise NotImplementedError()

def get_hamming_distance(state, goal):
    return sum(u!=v for u, v in zip(state, goal) if u!=0)

def heuristic_cost(state, goal, rank, method):
    if method == "LinearConflict":
        return get_conflict_manhattan_distance(state, goal, rank)
    elif method == 'ManHattan':
        return get_manhattan_distance(state, goal, rank)
    elif method == 'Hamming':
        return get_hamming_distance(state, goal)
    elif method == "BFS":
        return 0
    else:
        raise ValueError("expected 'LinearConflict' or 'ManHattan' or 'Hamming' or 'BFS'")

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

def solve(start:list, goal:list, heuristic='ManHattan'):
    frontier = PriorityQueue()
    frontier.put(PrioritizedItem(0, start))
    came_from = dict()
    came_from[tuple(start)] = None
    cost_so_far = dict()
    cost_so_far[tuple(start)] = 0

    rank = int(sqrt(len(goal)))

    while not frontier.empty():
        current = frontier.get().item

        if is_equal(current, goal):
            break
            
        for next in get_neighbors(current):
            new_cost = cost_so_far[tuple(current)] + 1
            if (tuple(next) not in cost_so_far) or (new_cost < cost_so_far[tuple(next)]):
                cost_so_far[tuple(next)] = new_cost
                priority = new_cost + heuristic_cost(next, goal, rank, method=heuristic)
                frontier.put(PrioritizedItem(priority, next))
                came_from[tuple(next)] = tuple(current)
    
    if tuple(goal) not in came_from:
        raise ValueError('Initial state cannot be transformed into goal state.')
    
    chains = []
    node = tuple(goal)
    while node is not None:
        chains.append(node)
        node = came_from[node]
    chains.reverse()
    return chains

def print_state(state):
    rank = int(sqrt(len(state)))
    word_width = len(str(rank * rank - 1))
    line_width = (word_width + 3) * rank + 1

    print('*' * line_width)
    _ = '* ' + ' '*word_width + ' '
    for i in range(rank):
        print(_ * rank + '*')
        for j in range(rank):
            print('* ' + str(state[i * rank + j]).center(word_width) + ' ', end='')
        print('*')
        print(_ * rank + '*')
        print('*' * line_width)

def print_chains(chains):
    print('chains:')
    print_state(chains[0])
    for state in chains[1:]:
        print('==>')
        print_state(state)  
