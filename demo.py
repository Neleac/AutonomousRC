# Sample code to demonstrate steering and throttle

import time
from pwm_controller import PWMController
import numpy as np

if __name__ == '__main__':
    # init pwm controller
    control = PWMController()

    '''
    # gradually steer left, gradually steer right
    control.steer(0)
    control.steer(0.5)
    control.steer(1)
    control.steer(0)
    control.steer(-0.5)
    control.steer(-1)
    control.steer(0)
    '''
    # gradually increase throttle
    # reverse throttle is not fully supported
    control.drive(0)
    control.drive(0.25)
    control.drive(0.5)
    control.drive(0.75)
    control.drive(1)
    control.drive(0)



