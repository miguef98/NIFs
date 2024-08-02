import numpy as np
import os

def load( filepath ):
    with open(filepath, 'r') as file:
        vertices = []
        vertex_normals = []
        for line in file:
            
            values = line.replace('\n','').split(' ')

            if values[0] == 'v':
                if len(values) != 4:
                    raise Exception('OBJ incorrect format')
                
                vertices.append( [ float(values[1]), float(values[2]), float(values[3])] )

            elif values[0] == 'vn':
                if len(values) != 4:
                    raise Exception('OBJ incorrect format')
                
                vertex_normals.append( [ float(values[1]), float(values[2]), float(values[3])] )

    return np.array(vertices), np.array(vertex_normals)

