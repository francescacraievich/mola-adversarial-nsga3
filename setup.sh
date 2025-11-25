#!/bin/bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

ros2 daemon stop && ros2 daemon start