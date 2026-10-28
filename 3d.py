from arm_demo import analyse_pose
import numpy as np
import trimesh

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MODEL_PATH = "robot_arm.stl"

#Load the model once
mesh = trimesh.load_mesh(MODEL_PATH)

vertices = mesh.vertices.copy()
faces = mesh.faces

print("Number of vertices:", len(vertices))
print("Number of triangles:", len(faces))
print("Model dimensions:", mesh.extents)

#Rotate verticies around the Z-axis
def rotate_vertices(vertices, angle_deg, pivot):
    angle = np.radians(angle_deg)

    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1],
    ])

    return (vertices - pivot) @ rotation.T + pivot