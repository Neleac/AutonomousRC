import time
import Adafruit_PCA9685

'''
Representation Invariant:
    PWM Pulses > 0
Abstraction Function:
    PWMController => a pulse-width-modulation motor controller for PCA9685 boards
'''
class PWMController:

    def __init__(self,
                 address = 0x40,
                 frequency = 60,
                 busnum = None,
                 init_delay = 0.1,
                 left_pulse = 250,
                 right_pulse = 450,
                 max_pulse = 490,
                 min_pulse = 300,
                 zero_pulse = 350):

        self.default_freq = 60
        self.pwm_scale = frequency / self.default_freq
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(frequency)
        time.sleep(init_delay)
        
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        
        '''
        # send zero pulse to calibrate ESC
        self._set_pulse(0, max_pulse)
        time.sleep(0.01)
        self._set_pulse(0, min_pulse)
        time.sleep(0.01)
        self._set_pulse(0, zero_pulse)
        time.sleep(1)
        '''

    
    def _set_pulse(self, channel, pulse):
        self.pwm.set_pwm(channel, 0, int(pulse * self.pwm_scale))


    # Linear mapping between two ranges of values
    def _map_range(self, x, X_min, X_max, Y_min, Y_max):
        X_range = X_max - X_min
        Y_range = Y_max - Y_min
        XY_ratio = float(X_range) / Y_range
        y = ((x - X_min) / XY_ratio + Y_min) // 1
        return int(y)


    def steer(self, angle):
        MIN_ANGLE = -1
        MAX_ANGLE = 1
        if angle < MIN_ANGLE or angle > MAX_ANGLE:
            print("Steering angle must be between -1 and 1")
            return

        pulse = self._map_range(angle, 
                                MIN_ANGLE, MAX_ANGLE,
                                self.left_pulse, self.right_pulse)

        self._set_pulse(1, pulse)
        time.sleep(1e-3)
        

    def drive(self, throttle):
        MIN_THROTTLE = -1
        MAX_THROTTLE =  1
        if throttle < MIN_THROTTLE or throttle > MAX_THROTTLE:
            print("Throttle must be between -1 and 1")
            return

        if throttle > 0:
            pulse = self._map_range(throttle,
                                    0, MAX_THROTTLE,
                                    self.zero_pulse, self.max_pulse)
        else:
            pulse = self._map_range(throttle,
                                    MIN_THROTTLE, 0,
                                    self.min_pulse, self.zero_pulse)

        self._set_pulse(0, pulse)
        time.sleep(1e-3)