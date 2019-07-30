# Sample code to demonstrate steering and throttle

import time
from pwm_controller import PWMController


if __name__ == '__main__':
    # init pwm controller
    control = PWMController()
    
    # set steering to center, left, right, center
    control.steer(0)
    time.sleep(1)
    control.steer(1)
    time.sleep(1)
    control.steer(-1)
    time.sleep(1)
    control.steer(0)
    time.sleep(1)
    
    # set throttle to backward, 0, forward, 0
    control.drive(0.3)  # doesn't actually run, not sure why, warmup?
    time.sleep(1)
    control.drive(0)
    time.sleep(1)
    control.drive(1)
    time.sleep(1)
    control.drive(0)
    time.sleep(1)
    control.drive(-1)
    time.sleep(1)
    control.drive(0)