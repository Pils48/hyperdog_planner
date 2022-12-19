#!/usr/bin/env python3

from __future__ import print_function

import sys
import rospy
import time
import numpy as np
from random import random
import matplotlib.pyplot as plt
from hyperdog_planner.srv import plan_request


def mean_smoothing(trajectory):
    window_size = 5
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    # Loop through the array t o
    #consider every window of size 3
    while i < len(trajectory) - window_size + 1:
        # Calculate the average of current window
        window_average = [round(np.sum(trajectory[i:i+window_size, 0]) / window_size, 2), round(np.sum(trajectory[i:i+window_size, 1]) / window_size, 2)]
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by one position
        i += 1
    moving_averages.append([trajectory[-1, 0], trajectory[-1, 1]])
    moving_averages = np.array(moving_averages)
    return moving_averages


def animate_trajectory(trajectory):
    plt.ion()
    fig = plt.figure()
    trajectory = np.array([[trajectory[i].x, trajectory[i].y] for i in range(len(trajectory))])
    trajectory = mean_smoothing(trajectory)
    ax = fig.add_subplot(111)
    ax.plot(trajectory[:, 0], trajectory[:, 1])
    ax.set_xlim(left=0, right=10)
    ax.set_ylim(bottom=0, top=6)
    triangular = np.array([[0,  -.1],
                        [0, .1],
                        [.4, 0], 
                        [0, -.1]]).T
    initial_tringular = triangular
    line1 = ax.plot(triangular.T[:, 0], triangular.T[:, 1])[0]
    theta = 0
    for i, (x_traj, y_traj) in  enumerate(trajectory):
        x_base, y_base = x_traj, y_traj
        if i != 0:
            x = np.array([1, 0])
            v = np.array([trajectory[i, 0] - trajectory[i-1, 0], 
                        trajectory[i, 1] - trajectory[i-1, 1]])
            theta = np.arccos(x.dot(v) / np.sqrt(v[0]**2 + v[1]**2))
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
        triangular = rotMatrix @ initial_tringular
        line1.set_xdata(triangular.T[:, 0] + x_base)
        line1.set_ydata(triangular.T[:, 1] + y_base)
        time.sleep(0.1)
        fig.canvas.draw()
        fig.canvas.flush_events()
    


def plan_trajectory():
    rospy.wait_for_service('plan_trajectory')
    try:
        plan_command = rospy.ServiceProxy('plan_trajectory', plan_request)
        planner_res = plan_command()
        if planner_res.success:
            animate_trajectory(planner_res.trajectory)
        return planner_res.success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    print("Requesting to plan trajectory from planner...")
    print(plan_trajectory())