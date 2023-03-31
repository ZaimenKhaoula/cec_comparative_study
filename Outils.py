import math
from shapely.geometry import Point, LineString
import networkx as nx
import Connectivity_repair_SP as sp






def distance(ax, ay, bx, by):
    return math.sqrt(pow(ax - bx, 2) + pow(ay - by, 2))

def obstacle_to_line(obstacle):
    return LineString([(obstacle.x0, obstacle.y0), (obstacle.x1, obstacle.y1)])

def generate_coordinates_archi2():
    """
    archi2 is composed of two rectangulars (40 X 30) and (20 X 35)
    """
    coordinates=[]
    for j in range(30):
        for i in range(60):
            coordinates.append([i + 0.5, j+0.5])

    for j in range(5):
        for i in range(20):
            coordinates.append([i + 40.5, j+30.5])

    return coordinates


def generate_coordinates_archi1():

    coordinates=[]
    for j in range(5):
        for i in range(15):
            coordinates.append([i + 0.5, j+0.5])

    for j in range(2):
        for i in range(5):
            coordinates.append([i + 0.5, j+5.5])

    return coordinates

def generate_coordinates_archi3():

    coordinates=[]
    for j in range(12):
        for i in range(20):
            coordinates.append([i + 0.5, j+0.5])

    for j in range(3):
        for i in range(5):
            coordinates.append([i + 0.5, j+12.5])

    for j in range(3):
        for i in range(5):
            coordinates.append([i+15.5, j+12.5])

    return coordinates

def generate_coordinates_archi4():

    coordinates=[]
    for j in range(15):
        for i in range(12):
            coordinates.append([i + 0.5, j+0.5])

    return coordinates

def generate_coordinates_archi5():

    coordinates=[]
    for j in range(20):
        for i in range(26):
            coordinates.append([i + 0.5, j+0.5])

    return coordinates

def generate_coordinates_archi6():

    coordinates=[]
    for j in range(10):
        for i in range(22):
            coordinates.append([i + 0.5, j+0.5])

    for j in range(6):
        for i in range(18):
            coordinates.append([i + 0.5, j+10.5])
    return coordinates



def generate_coordinates_archi7():

    coordinates=[]
    for j in range(7):
        for i in range(10):
            coordinates.append([i + 5.5, j+0.5])

    for j in range(23):
        for i in range(15):
            coordinates.append([i + 0.5, j+7.5])

    return coordinates


def generate_coordinates_archi8():

    coordinates=[]
    for j in range(5):
        for i in range(30):
            coordinates.append([i + 0.5, j+0.5])

    for j in range(20):
        for i in range(25):
            coordinates.append([i + 5.5, j+5.5])

    for j in range(5):
        for i in range(30):
            coordinates.append([i + 0.5, j+25.5])


    return coordinates



def generate_coordinates_archi9():

    coordinates=[]
    for j in range(13):
        for i in range(10):
            coordinates.append([i + 10.5, j+0.5])

    for j in range(5):
        for i in range(20):
            coordinates.append([i + 0.5, j+13.5])

    return coordinates


def generate_coordinates_archi10():

    coordinates=[]
    for j in range(10):
        for i in range(20):
            coordinates.append([i + 50.5, j+0.5])

    for j in range(20):
        for i in range(70):
            coordinates.append([i + 0.5, j+10.5])

    return coordinates