import numpy as np
from mealpy.swarm_based import PSO
from mealpy.utils.problem import Problem
from mealpy.evolutionary_based.GA import BaseGA
import Connectivity_repair_SP as sp
import networkx as nx
import random
import time
import Outils
from math import sqrt
import File_processing as fp
import PropagationModel as pm
from mealpy.swarm_based.BA import OriginalBA
from mealpy.evolutionary_based.DE import BaseDE
from mealpy.music_based.HS import BaseHS
from mealpy.swarm_based.CSA import OriginalCSA
from mealpy.swarm_based.ABC import OriginalABC
from mealpy.swarm_based.GWO import OriginalGWO
#index of zones far from the origin by D or less
def save_results(fichier, result):
    with open(fichier, 'a') as f:
        # Write a new line to the file
        f.write(result)
        f.write('\n')

vector_time=[]
nb_runs= 1
Rc = 10
Rs= 6
Ru=4
#nb_zones=20*35+40*30
nb_zones=20*12+5*2*3
pop_size= 100
epoch= 350
nb_targets =94
sensitivity= -96
file1 = open('targets_archi3', 'r')
list_target_points=[int(x) for x in file1.readline().split(',')]
t= time.time()
print("time to generate neigb per zone ", end=" ")
print(time.time()-t)
print("list connections between positions generated")
print("time to create graph", end=" ")
obstacles= fp.load_obstacles("archi3")
print("finish loading obstacles")
coordinates= Outils.generate_coordinates_archi3()
print("finish generating coordinates")
t= time.time()
graph = sp.generate_list_connections_between_positions(coordinates, obstacles,sensitivity, Rc)
print(time.time()-t)
print("finish generating graph of zones")

def create_deployment_graph(solution):
    index_of_deployed_sensors = [i for i in range(len(solution)) if solution[i] == 1]
    deployment_graph = graph.subgraph(index_of_deployed_sensors).copy()

    return deployment_graph
def covering_zones_for_each_target():
    result=[]
    for target in list_target_points:
        lst=[]
        for i in range(len(coordinates)):
            if pm.Elfes_model(coordinates[i][0],coordinates[i][1],coordinates[target][0], coordinates[target][1], Rs, Ru):
            #if pm.MWM(coordinates[i][0],coordinates[i][1],coordinates[target][0], coordinates[target][1], obstacles):
                lst.append(i)
        result.append(lst)

    return result


target_covering_zones= covering_zones_for_each_target()

def amend_position(self,solution, lb=None, ub=None):

    bins = [0.5]
    solution = np.digitize(solution, bins)
    deployment_graph= create_deployment_graph(solution)
    disjoint_sets = sp.distinct_connected_components(deployment_graph)
    if len(disjoint_sets) > 1:
        sp.connectivity_repair_heuristic(disjoint_sets, solution,  graph, coordinates)
    return np.array(solution)

def covered(target_point, solution):
    for i in range(len(solution)):
        if solution[i] == 1 and i in target_covering_zones[target_point]:
            return True
    return False


def calculate_cost(solution):
    return np.count_nonzero(solution)


def calculate_coverage(solution):
    coverage = 0
    for i in range(len(list_target_points)):
        if covered(i, solution):
            coverage = coverage + 1
    return coverage


def generate_random_solution(self,lb=None, ub=None):
    #print("start generating random solutions")
    solution=[0,] * nb_zones
    for i in range(nb_zones):
        if random.uniform(0, 1) > 0.75:
            solution[i] = 1
    #print(solution)
    return solution


def fit_func(solution):
    coverage=calculate_coverage(solution)
    cost= calculate_cost(solution)
    return (nb_targets - coverage) / nb_targets + cost / nb_zones


Problem.generate_position=generate_random_solution
Problem.amend_position= amend_position
Problem.fit_func=fit_func

deployment_problem= {
  "fit_func": fit_func,
  "lb": [0,] * nb_zones,
  "ub": [1,] *nb_zones,
  "minmax": "min",
  "log_to": None,
  "save_population": True,
}


for i in range(nb_runs):
    print("run number", end="  ")
    print(i)




    print("executing  BA")
    loudness = 0.8
    pulse_rate = 0.95
    pf_min = 0.
    pf_max = 10.
    model = OriginalBA(epoch, pop_size, loudness, pulse_rate, pf_min, pf_max)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution:{list(best_position)}, Fitness: {best_fitness}")
    #save_results('BA_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)

    """
    model.history.save_global_best_fitness_chart(filename="BA_global_best")
    model.history.save_runtime_chart(filename="BA_runtime_Chart")
    model.history.save_exploration_exploitation_chart(filename="BA_exploration")
    model.history.save_diversity_chart(filename="BA_diversity")
    """
    print("executing  PSO")
    model = PSO.OriginalPSO(epoch, pop_size, c1=2.0, c2=2, w_min=0.3, w_max=0.8)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}")
    #save_results('PSO_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)
    """
    model.history.save_global_best_fitness_chart(filename="PSO_global_best")
    model.history.save_runtime_chart(filename="PSO_runtime_Chart")
    model.history.save_exploration_exploitation_chart(filename="PSO_exploration")
    model.history.save_diversity_chart(filename="PSO_diversity")
    """
    print("executing  DE")
    wf = 0.7
    cr = 0.9
    strategy = 0
    model = BaseDE(epoch, pop_size, wf, cr, strategy)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution: {list(best_position)}, Fitness: {best_fitness}")
    #save_results('DE_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)
    """
    model.history.save_global_best_fitness_chart(filename="DE_global_best")
    model.history.save_runtime_chart(filename="DE_runtime_Chart")
    model.history.save_exploration_exploitation_chart(filename="DE_exploration")
    model.history.save_diversity_chart(filename="DE_diversity")
    """
    print("executing  HS")
    c_r = 0.95
    pa_r = 0.05
    model = BaseHS(epoch, pop_size, c_r, pa_r)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution: {list(best_position)}, Fitness: {best_fitness}")
    #save_results('HS_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)
    """
    model.history.save_global_best_fitness_chart(filename="HS_global_best")
    model.history.save_runtime_chart(filename="HS_runtime_Chart")
    model.history.save_exploration_exploitation_chart(filename="HS_exploration")
    model.history.save_diversity_chart(filename="HS_diversity")
    """

    print("executing GA")
    model = BaseGA(epoch, pop_size, pc=0.95, pm=0.1)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution: {list(best_position)}, Fitness: {best_fitness}")
    save_results('GA_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)

    print("executing  CSA")
    p_a = 0.3
    model = OriginalCSA(epoch, pop_size, p_a)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution: {list(best_position)}, Fitness: {best_fitness}")
    #save_results('CSA_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)
    """
    model.history.save_global_best_fitness_chart(filename="CSA_global_best")
    model.history.save_runtime_chart(filename="CSA_runtime_Chart")
    model.history.save_exploration_exploitation_chart(filename="CSA_exploration")
    model.history.save_diversity_chart(filename="CSA_diversity")
    """
    print("executing  GWO")
    model = OriginalGWO(epoch, pop_size)
    best_position, best_fitness = model.solve(deployment_problem)
    print(f"Solution: {list(best_position)}, Fitness: {best_fitness}")
    #save_results('GWO_archi3_correct_1', str(f"Best solution: {list(best_position)}, Best fitness: {best_fitness}"))
    print(model.history.list_global_best_fit)





