class Obstacle:
    "type is a boolean variable. type=True if it is outer wall, False otherwise "
    def __init__(self, x0, y0, x1, y1,type):
        self.x0=x0
        self.x1 = x1
        self.y0=y0
        self.y1 = y1
        self.type=type

def load_obstacles(fichier):
    obstacles=[]
    with open(fichier, 'r') as file:
        for line in file:
            words = line.split(',')
            o=Obstacle(int(words[0]),int(words[1]),int(words[2]),int(words[3]),int(words[4])==0)
            obstacles.append(o)
    return obstacles

def save_results(fichier, result):
    with open(fichier, 'a') as f:
        # Write a new line to the file
        f.write(result)
        f.write('\n')