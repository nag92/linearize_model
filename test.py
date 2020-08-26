from ilqr.dynamics import FiniteDiffDynamics
import numpy as np
state_size = 2  # [position, velocity]
action_size = 1  # [force]

dt = 0.01  # Discrete time-step in seconds.
m = 1.0  # Mass in kg.
alpha = 0.1  # Friction coefficient.

def f(x, u, i):
    """Dynamics model function.

    Args:
        x: State vector [state_size].
        u: Control vector [action_size].
        i: Current time step.

    Returns:
        Next state vector [state_size].
    """
    [x, x_dot] = x
    [F] = u

    # Acceleration.
    x_dot_dot = x_dot * (1 - alpha * dt / m) + F * dt / m
    print
    return np.array([
        x + x_dot * dt,
        x_dot + x_dot_dot * dt,
    ])

# NOTE: Unlike with AutoDiffDynamics, this is instantaneous, but will not be
# as accurate.
dynamics = FiniteDiffDynamics(f, state_size, action_size)
print(dynamics.action_size)