#!/usr/bin/python3

from queue import LifoQueue, Queue
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        
        while not foundTour and time.time()-start_time < time_allowance:
            route = []
            randomNumber = random.randint(0, ncities-1)
            start_city = cities[randomNumber]
            current_city = start_city
            route.append(current_city)
            while(True):
                minimum_length = math.inf
                next_city = None
                for city in cities:
                    if(current_city.costTo(city) < minimum_length):
                        if(route.count(city) < 1):
                            minimum_length = current_city.costTo(city)
                            next_city = city
                if(next_city == None):
                    if(current_city.costTo(start_city) != math.inf and len(route) == ncities):
                        foundTour = True
                    break
                route.append(next_city)
                current_city = next_city

        if(foundTour):
            bssf = TSPSolution(route)
        
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

    def branchAndBound(self, time_allowance=60.0):
        # Get initial bssf, currently using random but if
        # I have the time I'll change it to greedy
        results = {}
        cities = self._scenario.getCities()
        random_result = self.greedy()
        bssf_cost = random_result['cost']
        bssf_route = random_result['soln']

        # Initialize needed values
        queue = []
        solution_count = 0
        max_queue_size = 0
        current_queue_size = 0
        minimum_cost = 0
        pruned = 0
        nodes_generated = 0
        ncities = len(cities)
        reduced_cost_matrix = np.empty((ncities, ncities))
        for i in range(0, ncities):
            for j in range(0, ncities):
                reduced_cost_matrix[i][j] = cities[i].costTo(cities[j])

        # creates initial reduced cost matrix
        reduced_cost_matrix, minimum_cost = self.minimizeMatrix(reduced_cost_matrix)
        start_time = time.time()

        #Generate initial state with first city
        starting_city = cities[0]
        init_state = State(reduced_cost_matrix, np.array([starting_city]), minimum_cost, starting_city._index)

        #Initialize queue with first state
        heapq.heappush(queue, init_state)
        current_queue_size += 1
        max_queue_size = 1

        #Begin searching
        while (len(queue) != 0 and time.time()-start_time < time_allowance):

            #Pop next possible solution off queue
            current_state = heapq.heappop(queue)
            current_queue_size -= 1

            #Check if current state is still good
            if(current_state.cost < bssf_cost):

                #Start creating next potential states
                for city in cities:

                    #If the city has already been visited,
                    #don't create a state
                    if city not in current_state.path:

                        #If there is no connecting edge, go to next city
                        distance = current_state.matrix[current_state.city_index, city._index]
                        if(distance == math.inf):
                            continue

                        #Create a substate with every valid city
                        nodes_generated += 1

                        #Copy the previous state's matrix
                        new_matrix = np.copy(current_state.matrix)

                        #Reduce the copied matrix
                        new_matrix[:,city._index] = math.inf
                        new_matrix[current_state.city_index,:] = math.inf
                        new_matrix[city._index, current_state.city_index] = math.inf
                        new_matrix, cost_to_adjust = self.minimizeMatrix(new_matrix)

                        #Copy the path from the previous state
                        #Add the current state's city to its path
                        new_path = np.copy(current_state.path)
                        new_path = np.append(new_path, city)

                        #Calculate the lowerbound of the current state
                        new_score = current_state.cost + distance + cost_to_adjust

                        #Create state object
                        new_state = State(new_matrix, new_path, new_score, city._index)

                        #Check if it is a complete path
                        if(len(new_state.path) == ncities and city.costTo(starting_city) != math.inf):

                            #Update bssf if needed
                            if(new_state.cost < bssf_cost):
                                bssf_route = TSPSolution(new_state.path)
                                bssf_cost = bssf_route.cost
                                solution_count +=1
                                
                        #If the path is not complete, 
                        #add or prune this state
                        elif new_state.cost < bssf_cost:
                            heapq.heappush(queue, new_state)
                            current_queue_size += 1
                            if(current_queue_size > max_queue_size):
                                max_queue_size = current_queue_size
                        else:
                            pruned += 1



        end_time = time.time()
        results['cost'] = bssf_cost
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = bssf_route
        results['max'] = max_queue_size
        results['total'] = nodes_generated
        results['pruned'] = pruned

        return results

    def minimizeMatrix(self, matrix):
        new_matrix = np.copy(matrix)

        min_rows = new_matrix.min(1)
        n = len(min_rows)

        minimum_cost = 0
        for i in range(n):
            if(min_rows[i] != math.inf):
                minimum_cost += min_rows[i]

        for i in range(n):
            for j in range(n):
                if new_matrix[i][j] != math.inf:
                    new_matrix[i][j] -= min_rows[i]

        min_cols = new_matrix.min(0)

        for i in range(n):
            if(min_cols[i] != math.inf):
                minimum_cost += min_cols[i]

        for i in range(n):
            for j in range(n):
                if new_matrix[j][i] != math.inf:
                    new_matrix[j][i] -= min_cols[i]

        return new_matrix, minimum_cost

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        pass


class State:
    def __init__(self, matrix, path, score, index):
        self.matrix = matrix
        self.path = path
        self.cost = score
        self.city_index = index

    def __lt__(self, other):
        
        return self.cost / len(self.path) < other.cost / len(other.path)