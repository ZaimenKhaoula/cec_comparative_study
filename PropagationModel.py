import math
from shapely.geometry import LineString
import Outils
from scipy.spatial import distance
import math

def Elfes_model(xs, ys, xT, yT, Rs, Ru):
    x=distance.euclidean([xs, ys], [xT, yT])
    if x<= Ru:
        return True
    else:
        if x> Ru and x<= Rs-Ru and math.exp(x-Ru)* 0.01 <0.5:
            return True
        else:
            return False


def MWM(xs1, ys1, xs2, ys2, obstacles,threshold, rc):
    Pt=3
    d= distance.euclidean([xs1, ys1], [xs2, ys2])
    f= 2400000000
    L=1
    PLd0=7.5
    n = 2.8
    interference_1= -6 #for plasterboard
    interference_2 = -25 # for concrete
    interference_3=-10 #for brick
    # Calculate the loss due to walls
    L_mw = 0
    for obstacle in obstacles:
        if Outils.obstacle_to_line(obstacle).intersects(LineString([(xs1, ys1), (xs2, ys2)])):
            if obstacle.type == 1:
                L_mw = L_mw + interference_1
            else:
                L_mw = L_mw + interference_2


    # Calculate the signal strength at the receiver
    PL = Pt- PLd0- 10 * n * math.log10(d+0.01) + L_mw

    return PL>=threshold and d<=rc



def COSTA213(xs1, ys1, xs2, ys2, obstacles):

    """
    Calculate the signal strength of a wireless signal using the COST231 multiwall model.

    Parameters:
    - Pt: transmitter power
    - d: distance between the transmitter and the receiver (in meters)
    - f: frequency of the signal (in Hz)
    - h_t: height of the transmitter antenna (in meters)
    - h_r: height of the receiver antenna (in meters)
    - L: system loss factor (in dB)
    - alpha: path loss exponent (default value is 3.5)
    - beta: shadow fading standard deviation (default value is 0)
    - n: number of walls (default value is 1)
    - wall_type: type of wall (default value is 'brick')

    Returns:
    - The signal strength of the wireless signal (in dBm)
    """
    Pt=1
    d= distance.euclidean([xs1, ys1], [xs2, ys2] )
    f= 2400000000
    h_t=1
    h_r=1
    L=1
    alpha = 3.5
    beta = 0
    n = 1
    wall_type = 'brick'
    # Calculate the wavelength of the signal
    lambda_ = 3e8 / f

    # Calculate the free space loss
    L_fs = 32.45 + 20 * math.log10(f) + 20 * math.log10(d)

    # Calculate the path loss
    L_pl = L_fs + 10 * alpha * math.log10(d)

    # Calculate the effect of antenna height
    L_ah = 0
    if h_t > 1 and h_r > 1:
        L_ah = (1.1 * math.log10(f) - 0.7) * h_t - (1.1 * math.log10(f) - 0.7) * h_r
        L_ah += 20 * math.log10(h_t) + 20 * math.log10(h_r)

    # Calculate the shadow fading
    L_sf = beta * math.sqrt(d)

    # Calculate the loss due to walls
    L_mw = 0
    for obstacle in obstacles:
        if Outils.obstacle_to_line(obstacle).intersects(LineString([(xs1, ys1), (xs2, ys2)])):
            if obstacle.type == 1:
                L_mw = L_mw + 10 * n * math.log10(d)
            else:
                L_mw = L_mw + 15 * n * math.log10(d)

    PLd0=1
    # Calculate the signal strength at the receiver
    PL = PLd0 - L_mw

    return PL