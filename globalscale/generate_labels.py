import os
import numpy as np
import shutil
import pickle
import networkx as nx
import cv2

IMAGE_SIZE = 2048
KEYPOINT_RADIUS = 3
ROAD_WIDTH = 3
# Load GT Graph
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory where script lives
DATA_DIR = os.path.join(BASE_DIR, "Global-Scale")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")


def create_directory(dir, delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

#image_deg,size=IMAGE_SIZE, degree =key_degree,points=key_nodes, radius=KEYPOINT_RADIUS
def draw_points_on_image(size, degrees,points, radius):
    """
    Draws points on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - points: A list of tuples, where each tuple represents the (x, y) coordinates of a point in pixel coordinates.
    - radius: The radius of the circles to be drawn for each point, in pixels.

    Returns:
    - A square image with the given points drawn as filled circles.
    """

    # Create a square image of the given size, initialized to zeros (black), with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of points
    for point,degree in zip(points,degrees):
        # Draw each point as a filled circle on the image
        # The circle is drawn with center at 'point', radius as specified, color 255 (white), and filled (thickness=-1)
        cv2.circle(image, point, radius, 255, -1)
        #cv2.circle(image_deg,point,radius,degree,-1)

    return image


def draw_line_segments_on_image(size, line_segments, width):
    """
    Draws line segments on a square image using OpenCV.

    Parameters:
    - size: The size of the square image (width and height) in pixels.
    - line_segments: A list of tuples, where each tuple represents a line segment as ((x1, y1), (x2, y2)).
    - width: The width of the lines to be drawn, in pixels.

    Returns:
    - A square image with the given line segments drawn.
    """

    # Create a square image of the given size, initialized to zeros (black)
    # with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of line segments
    for segment in line_segments:
        # Unpack the start and end points of the line segment
        (x1, y1), (x2, y2) = segment

        # Draw the line segment on the image
        # The line is drawn with color 255 (white) and the specified width
        #cv2.line(image, (x1, y1), (x2, y2), 51, 5)
        #cv2.line(image, (x1, y1), (x2, y2), 153, 3)
        cv2.line(image, (x1, y1), (x2, y2), 255, 3)
        #cv2.line(demask,(x1, y1), (x2, y2), 2, 5)

    return image

GLOBALSCALE_DIRS = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
] # adds the folders inside of Global-Scale

OUTPUT_DIR_PROCS = []
for DIR in GLOBALSCALE_DIRS:
    OUTPUT_DIR_PROC = os.path.join(OUTPUT_DIR, DIR) # Separate processed folders as labeled by the different dataset subdivs
    OUTPUT_DIR_PROCS.append(OUTPUT_DIR_PROC)
    create_directory(OUTPUT_DIR_PROC, delete=True)

for i, DIR in enumerate(GLOBALSCALE_DIRS):
    print("\n")
    print(f"----------PROCESSING '{DIR}' FOLDER----------")
    DATA_DIR_REC = os.path.join(DATA_DIR, DIR) # Accessing the specific folder within "Global-scale"
    files = sorted(
        [f for f in os.listdir(DATA_DIR_REC)
        if f.endswith("_refine_gt_graph.p")],
        key=lambda x: int(x.split("_")[1])
    ) # gets the exact number of files to avoid FileNotFoundError and sorts them by tile index
    
    for fname in files:
        tile_index = fname.split("_")[1]
        print(f'Processing globalscale {DIR} tile {tile_index}.')
        vertices = []
        edges = []
        vertex_flag = True

        

        gt_graph = pickle.load(open(os.path.join(DATA_DIR_REC, f"region_{tile_index}_refine_gt_graph.p"), 'rb'))
        graph = nx.Graph()  # undirected
        for n, neis in gt_graph.items():
            for nei in neis:
                graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))

        # Collect key nodes (degree != 2)
        key_nodes = []
        key_degree = []
        for node, degree in graph.degree():


            if degree != 2:
                key_nodes.append(node)
                key_degree.append(degree)

        # Create key point mask
        # image_deg = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        #Create road mask
        road_mask = draw_line_segments_on_image(
            size=IMAGE_SIZE, line_segments=graph.edges(), width=ROAD_WIDTH)

        keypoint_mask = draw_points_on_image( size=IMAGE_SIZE, degrees=key_degree, points=key_nodes,
                                                    radius=KEYPOINT_RADIUS)
        #print(degree_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR_PROCS[i], f'keypoint_mask_{tile_index}.png'), keypoint_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR_PROCS[i], f'road_mask_{tile_index}.png'), road_mask)
        #cv2.imwrite(os.path.join(OUTPUT_DIR, f'degree_mask_{tile_index}.png'), degree_mask)

