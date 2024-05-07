# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi

deg2rad = pi/180.0

def makeWaypoints():
    
    v_average = 10

    t_ini = 3
    t = np.array([5, 5, 5, 5])
    
    wp_ini = np.array([0, 0, 0])
    wp = np.array([[0, 0, 10],
                   [0, 0, 20],
                   [0, 0, 30],
                   [0, 0, 40],
                   [0, 0, 50]])

    yaw_ini = 0    
    yaw = np.array([0, 0, 0, 0, 0])

    t = np.hstack((t_ini, t)).astype(float)
    wp = np.vstack((wp_ini, wp)).astype(float)
    yaw = np.hstack((yaw_ini, yaw)).astype(float)*deg2rad

    return t, wp, yaw, v_average
