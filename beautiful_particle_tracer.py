import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from numba import njit
import pandas as pd
import datashader as ds
from datashader.utils import export_image
import colorcet


CHARGE = 1
MASS = 1


@njit
def acceleration(x, v):
    """accelleration
    
    Parameters
    ----------
    x :: np.ndarray
        position 
    v :: np.ndarray
        velocity 

    Returns
    -------
    a :: np.ndarray

    """
    # rho_sq = np.sin(x[0]) + x[1] * x[2] + 1
    return - CHARGE * np.array([x[0], x[1], x[2]]) / MASS
    return - CHARGE * np.array([x[0] / rho_sq, x[1] / rho_sq, x[2] / rho_sq]) / MASS

@njit
def rk4(x0, v0, dt):
    """ Runge-kutta.

    Parameters
    ----------
    x0 :: np.ndarray
        position 
    v0 :: np.ndarray
        velocity 
    dt :: float
        time step

    Returns
    -------
    x :: np.ndarray, v :: np.ndarray
    

    Source
    ------
    http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
    """
    a0 = acceleration(x0, v0)

    x1 = x0 + 0.5 * v0 * dt
    v1 = v0 + 0.5 * a0 * dt
    a1 = acceleration(x1, v1)

    x2 = x0 + 0.5 * v1 * dt
    v2 = v0 + 0.5 * a1 * dt
    a2 = acceleration(x2, v2)

    x3 = x0 + v2 * dt
    v3 = v0 + a2 * dt
    a3 = acceleration(x3, v3)

    x4 = x0 + (dt / 6) * (v0 + 2*v1 + 2*v2 + v3)
    v4 = v0 + (dt / 6) * (a0 + 2*a1 + 2*a2 + a3)

    return x4, v4

def trajectory(t0, x0, v0, dt, max_iterations):
    """ step-by-step 3D particle trajectory

    Parameters
    ----------
    t0 :: float
        initial time
    x0 :: np.ndarray
        initial position 
    v0 :: np.ndarray
        initial velocity 
    dt :: float
        time step
    max_iterations :: int
        number of steps

    Returns
    -------
    np.array([['time', 'x', 'y', 'z']])
    """
    # initialise
    i = 1
    t = t0
    x = np.array(x0)
    v = np.array(v0)
    result = [[t, x[0], x[1], x[2]]]
    # step-by-step trajectory
    while i < max_iterations:
        x, v = rk4(x, v, dt)
        t += dt
        # record
        result.append([t, x[0], x[1], x[2]])
        # next step
        i += 1
    # output
    return result

def initialize(num, t0=0.0, x0=0.0, v0=0.0, sigma_t=None, sigma_x=None, sigma_v=None):
    """ 3D Gaussian time, position, and velocity distributions.
        
    Parameters
    ----------
    num :: int
        number of particles
    t0 :: float
        mean initial time
    x0 :: float, or tuple(float, float, float)
        mean initial position 
    v0 :: float, or tuple(float, float, float)
        mean initial velocity 
    sigma_t :: None or float
        time spread 
    sigma_x :: None, float, or tuple(float, float, float)
        position spread 
    sigma_v :: None, float, or tuple(float, float, float)
        velocity spread 

    Returns
    -------
    pd.DataFrame(columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    """
    num = int(num)
    # pandas DataFrame
    columns = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.DataFrame(columns=columns, index=np.arange(num), dtype='float64')
    # time
    df.time = t0
    if sigma_t is not None:
        df.time += np.random.randn(num) * sigma_t * 2.0**-0.5
    # position
    if not isinstance(x0, Iterable):
        x0 = (x0, x0, x0)
    if sigma_x is None:
        df.x = x0[0]
        df.y = x0[1]
        df.z = x0[2]
    else:
        if not isinstance(sigma_x, Iterable):
            sigma_x = (sigma_x, sigma_x, sigma_x)
        df.x = x0[0] + np.random.randn(num) * sigma_x[0] * 2.0**-0.5
        df.y = x0[1] + np.random.randn(num) * sigma_x[1] * 2.0**-0.5
        df.z = x0[2] + np.random.randn(num) * sigma_x[2] * 2.0**-0.5
    # velocity
    if not isinstance(v0, Iterable):
        v0 = (v0, v0, v0)
    if sigma_v is None:
        df.vx = v0[0]
        df.vy = v0[1]
        df.vz = v0[2]
    else:
        if not isinstance(sigma_v, Iterable):
            sigma_v = (sigma_v, sigma_v, sigma_v)
        df.vx = v0[0] + np.random.randn(num) * sigma_v[0] * 2.0**-0.5
        df.vy = v0[1] + np.random.randn(num) * sigma_v[1] * 2.0**-0.5
        df.vz = v0[2] + np.random.randn(num) * sigma_v[2] * 2.0**-0.5
    return df

def fly(initial, dt, **kwargs):
    """ Calculate trajectories
    
    Parameters
    ----------
    initial :: pd.DataFrame
        initial positions and velocities     
    dt :: float64
        time step                                       (s)
    max_iterations :: int

    tqdm_kw :: dict
        keyword arguments for tqdm (progress bar)
        
    Returns
    -------
    step-by-step trajectories for all particles ::
        pd.DataFrame(index=['particle', 'time'], columns=['x', 'y', 'z'])    
    """
    max_iterations = kwargs.get("max_iterations", int(1e5))
    tqdm_kw = kwargs.get("tqdm_kw", {})
    # fly ions
    num = len(initial.index)
    result = {}
    for i, row in tqdm(initial.iterrows(), total=num, **tqdm_kw):
        t0 = row.time
        x0 = np.array([row.x, row.y, row.z])
        v0 = np.array([row.vx, row.vy, row.vz])
        tr = pd.DataFrame(trajectory(t0, x0, v0, dt, max_iterations),
                          columns=['time', 'x', 'y', 'z']).set_index("time")
        result[i] = tr
    # output
    return pd.concat(result, names=["particle"])


def init_chain(num=16, t0=0):
    # pandas DataFrame
    columns = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.DataFrame(columns=columns, index=np.arange(num), dtype='float64')
    # time
    df.time = t0
    # position
    df.x = 0
    df.y = np.linspace(-1, 1, num)
    df.z = 0
    # velocity
    df.vx = 0
    df.vy = 0 # np.linspace(2, -2, num)
    df.vz = 0.1

    return df

# starting conditions
initial = init_chain()

# calc.
df = fly(initial, dt=.001, max_iterations=int(6e4))

cvs = ds.Canvas(plot_width=2000, plot_height=500, y_range=(-5, 5))
agg = cvs.points(df, 'z', 'y')
img = ds.tf.set_background(ds.tf.shade(agg, cmap=colorcet.rainbow), 'black')
ds.utils.export_image(img, 'trajectories')
