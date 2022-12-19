#!/usr/bin/env python3

from __future__ import print_function

import sys
import rospy
from hyperdog_planner.srv import plan_request


def plan_trajectory():
    rospy.wait_for_service('plan_trajectory')
    try:
        plan_command = rospy.ServiceProxy('plan_trajectory', plan_request)
        planner_res, success = plan_command()
        # print(planner_res.res.trajectory)
        return success
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    print("Requesting to plan trajectory from planner...")
    print(plan_trajectory())