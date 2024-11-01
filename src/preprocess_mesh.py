import numpy as np
import trimesh as tm
import os

def normalizeMesh( mesh ):
    center = np.mean( mesh.vertices, axis=0)
    T = np.block( [ [np.eye(3,3), -1 * center.reshape((3,1))], [np.eye(1,4,k=3)]])

    mesh.apply_transform( T )

    max_coord = np.max( np.abs( mesh.vertices ))

    S = np.block([ [np.eye(3,3) * (1 / (max_coord + max_coord * 0.1)), np.zeros((3,1))], [np.eye(1,4,k=3)] ]) 
    mesh.apply_transform( S )

    return mesh

def preprocessMesh( outputPath, meshFile, surfacePoints=1e5 ):
    mesh = tm.load_mesh(meshFile)
    mesh = normalizeMesh( mesh )

    vertices, faces = tm.sample.sample_surface( mesh, surfacePoints )

    bary = tm.triangles.points_to_barycentric(
        triangles=mesh.triangles[faces], points=vertices)
    # interpolate vertex normals from barycentric coordinates
    normals = tm.unitize((mesh.vertex_normals[mesh.faces[faces]] *
                              tm.unitize(bary).reshape(
                                  (-1, 3, 1))).sum(axis=1))

    #normals = mesh.face_normals[faces]
    #normals /= np.linalg.norm( normals , axis=1)[ ..., None ]

    mesh_name = meshFile[meshFile.rfind('\\') + 1 : meshFile.rfind('.') ]
    print(mesh_name)

    with open( os.path.join( outputPath, mesh_name + '_pc.obj'), 'w+') as output_file:
        for vertex in vertices:
            output_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

        for normal in normals:
            output_file.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')


    with open( os.path.join( outputPath, mesh_name + '_pc.ply'), 'w+') as output_file:
        output_file.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n"
        )
        for vertex, normal in zip(vertices, normals):
            output_file.write(f'{vertex[0]} {vertex[1]} {vertex[2]} {normal[0]} {normal[1]} {normal[2]}\n')
    