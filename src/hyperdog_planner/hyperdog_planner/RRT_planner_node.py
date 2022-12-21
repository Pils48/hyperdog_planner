#!/usr/bin/env python3

import time
import geometry_msgs.msg
import numpy as np
import rclpy
from random import random
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from collections import deque
from hyperdog_msgs.msg import JoyCtrlCmds


class Line():
    ''' Define line '''
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist # normalize

    def path(self, t):
        return self.p + t * self.dirn


def Intersection(line, center, radius):
    ''' Check line-sphere (circle) intersection '''
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a);
    t2 = (-b - np.sqrt(discriminant)) / (2 * a);

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True



def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def isInObstacle(vex, obstacles, radius):
    for obs in obstacles:
        if distance(obs, vex) < radius:
            return True
    return False


def isThruObstacle(line, obstacles, radius):
    for obs in obstacles:
        if Intersection(line, obs, radius):
            return True
    return False


def nearest(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def newVertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min (stepSize, length)

    newvex = (nearvex[0]+dirn[0], nearvex[1]+dirn[1])
    return newvex


def window(startpos, endpos):
    ''' Define seach window - 2 times of start to end rectangle'''
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def isInWindow(pos, winx, winy, width, height):
    ''' Restrict new vertex insides search window'''
    if winx < pos[0] < winx+width and \
        winy < pos[1] < winy+height:
        return True
    else:
        return False


class Graph:
    ''' Define graph '''
    def __init__(self, startpos, endpos):
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos:0}
        self.neighbors = {0:[]}
        self.distances = {0:0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


    def randomPosition(self):
        rx = random()
        ry = random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        return posx, posy


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):
    ''' RRT algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            #print('success')
            # break
    return G


def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize):
    ''' RRT star algorithm '''
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.randomPosition()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = newVertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx]+dist)
            except:
                G.distances[endidx] = G.distances[newidx]+dist

            G.success = True
            #print('success')
            # break
    return G



def dijkstra(G):
    '''
    Dijkstra algorithm for finding shortest path from start position to end.
    '''
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)



def plot(G, obstacles, radius, path=None):
    '''
    Plot RRT, obstacles and shortest path
    '''
    px = [x for x, y in G.vertices]
    py = [y for x, y in G.vertices]
    fig, ax = plt.subplots()

    for obs in obstacles:
        circle = plt.Circle(obs, radius, color='red')
        ax.add_artist(circle)

    ax.scatter(px, py, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i+1]) for i in range(len(path)-1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path

class Planner_Node():
    def __init__(self, node_name = 'hyperdog_planner'):
        rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        print("Planner node has been initialized")
        self.pub_name = 'hyperdog_joy_ctrl_cmd'
        self.pub_queueSize = 12
        self.pub_interface = JoyCtrlCmds   #hyperdog_msgs.msg.Geometry
        self.pub_timer_period = 0.01
        self.pub_callback = self._pub_callback
        self.pub_timer = self.node.create_timer(self.pub_timer_period, self.pub_callback)
        self.obstacles = []
        self.pub = self.node.create_publisher(
                            self.pub_interface,
                            self.pub_name,
                            self.pub_queueSize
                        )
        rclpy.spin(self.node)


    def form_control_msg(self, **kwargs):
        msg = JoyCtrlCmds()
        msg.states[0] = kwargs["start"]
        msg.states[1] = kwargs["walk"]
        msg.states[2] = kwargs["side_start"]
        msg.gait_type = kwargs["gait_type"] 
        pose = geometry_msgs.msg.Pose()
        pose.position.x = kwargs["x"]
        pose.position.y = kwargs["y"]
        pose.position.z = kwargs["height"]
        pose.orientation.w = kwargs["q_w"]
        pose.orientation.x = kwargs["q_x"]
        pose.orientation.y = kwargs["q_y"]
        pose.orientation.z = kwargs["q_z"]
        msg.pose = pose
        v = geometry_msgs.msg.Vector3()
        v.x = kwargs["slant_x"]
        v.y = kwargs["slant_y"]
        v.z = kwargs["step_height"]
        msg.gait_step = v
        return msg


    def _pub_callback(self):
        print("Sending trajectory...")
        ctrl_msg = self.form_control_msg(start=True, walk=True, side_start=False,
                                        gait_type=1, x=0.0, y=0.0, height=140.0, q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
                                        slant_x=150.0, slant_y=0.0, step_height=50.0)
        time.sleep(3) # Wait 10 seconds for execution TO DO: add wycon system for feedback
        self.pub.publish(ctrl_msg)
        ctrl_msg = self.form_control_msg(start=True, walk=True, side_start=False,
                                        gait_type=1, x=0.0, y=0.0, height=180.0, q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
                                        slant_x=0.0, slant_y=150.0, step_height=50.0)
        time.sleep(3)
        self.pub.publish(ctrl_msg)
        ctrl_msg = self.form_control_msg(start=True, walk=True, side_start=False,
                                        gait_type=1, x=0.0, y=0.0, height=140.0, q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
                                        slant_x=-150.0, slant_y=0.0, step_height=50.0)
        time.sleep(3)
        self.pub.publish(ctrl_msg)
        ctrl_msg = self.form_control_msg(start=True, walk=True, side_start=False,
                                        gait_type=1, x=0.0, y=0.0, height=180.0, q_w=1.0, q_x=0.0, q_y=0.0, q_z=0.0,
                                        slant_x=0.0, slant_y=-150.0, step_height=50.0)
        time.sleep(3)
        self.pub.publish(ctrl_msg)
      
    
    def addCollisionObject(self, obstacles):
        for obstacle in obstacles:
            self.obstacles.append()


def main():
    planner_node = Planner_Node()


if __name__ == '__main__':
    main()
