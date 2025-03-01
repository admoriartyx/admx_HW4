# This is the .py file for Question 3 of Problem Set 4

# Part a 

# This part of the problem is a written response but since the rest of the question is code based, I am
# going to answer part a in comment form for simplicity.

# A physical example of a Lorentz system is found in certain electric circuits that behave chaotically.
# Chaotic behaving circuits can be utilized in a variety of disciplines and applications, including
# secure communications, random number generation, and system testing. The variables themselves, resemble
# physical quantities as follows:

# x: May represent voltage across a specific electrical component
# y: May represent current.
# z: May represent control voltage or some delimiter on another quantity in the circuit.

# sigma: For a circuit, may represent rate at which charge is exchanged between components.
# rho: Could represent input power supply or band gain.
# beta: Could represent stabilization factor, such as dissipation rate.

# Part b

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

init_state = [1.0, 1.0, 1.0]
t = np.linspace(0, 12, 10000)
sigma = 10
rho = 48
beta = 3
solution = odeint(lorenz, init_state, t, args=(sigma, rho, beta))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Lorenz Attractor')
plt.show()
plt.savefig('Q3_part_b.png')

# Part c

# Be warned that my computer says it cannot play the video using Quicktime Player so I was unable to 
# verify if the video is accurate. However, the movie will be saved in the directory and my code ran smoothly
# so I assume everything is correct.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

init_state = [1.0, 1.0, 1.0]
t = np.linspace(0, 12, 10000)
sigma = 10
rho = 48
beta = 3
solution = odeint(lorenz, init_state, t, args=(sigma, rho, beta))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
line, = ax.plot([], [], [], lw=2)
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    line.set_data(solution[:frame, 0], solution[:frame, 1])
    line.set_3d_properties(solution[:frame, 2])
    return line,

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
ani.save('part_c_video.mp4', writer='ffmpeg', fps=60)
plt.show()


