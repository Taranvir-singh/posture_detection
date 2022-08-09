import random
import math
import numpy as np
def random_color():
	r = random.randint(0, 255)
	g = random.randint(0, 255)
	b = random.randint(0, 255)
	rand_color = (r, g, b)
	return rand_color
def findDistance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    dist = abs(math.sqrt((x2-x1)**2+(y2-y1)**2))
    return dist
def calculate_angle(a,b,c):
    angle = 0
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1],
    					 c[0]-b[0]) - np.arctan2(a[1]-b[1],
    					 a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle