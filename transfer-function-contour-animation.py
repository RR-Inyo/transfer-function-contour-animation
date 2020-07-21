# -*- coding: utf-8 -*-
"""
transfer-function-contour-animation.py
Copyright (c) 2020 @RR_Inyo
Released under the MIT license.
https://opensource.org/licenses/mit-license.php
"""

# Preparation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# Circuit and control parameters
# A proportional-and-integral controller for the current in an RL circuit
# with a transport delay and a moving average filter, of T for both
R = 20e-3 # [Ω]
L = 5e-3 # [h]
T = 100e-6 # [s]
K = 1
KP = K * L / (4 * T) # [V/A]
KI = KP * (R / L) # [V/As]

# Definition of transfer functions
## Forward elements
def G(s):
    return (KP + KI / s) * np.exp(-s * T) * (1 / (s * L + R)) # Sampled
    # return (KP + KI / s) * (1 / (s * L + R)) # Ideal

## Feedback elements
def H(s):
    T = 100e-6
    return (1 - np.exp(-s * T)) / (s * T) # Sampled
    # return 1 # Ideal

# Mesh grid preparation
## Range and resolution of drawing
scale = 20000
points = 100

## s = σ + jω
sigma = np.linspace(-scale, scale, points)
omega = np.linspace(-scale, scale, points)

## Generating the mesh grid
Sigma, Omega = np.meshgrid(sigma, omega)

# Time of starting for animation making
t0 = time.time()

# Initialization of the figure object
fig, ax = plt.subplots(1, 2, figsize = (16, 8))

# Animation parameters
N_FRAMES = 100
INTERVAL = 20
REPEAT_DELAY = 1000

# Animation updating function
def update(frame):
    # Clear the axes
    ax[0].cla()
    ax[1].cla()
    fig.patch.set_facecolor('lavender')
    
    # Parameter K
    global K, KP, KI
    K = (frame + 1) / N_FRAMES * 8
    KP = K * L / (4 * T) # [V/A]
    KI = KP * (R / L) # [V/As]
    
    # Calculating open-loop transfer function To(s) = G(s)H(s)
    ## Response to s = σ + jω, open-loop
    To_resp = G(Sigma + 1j * Omega) * H(Sigma + 1j * Omega)

    ## Magnitude (gain), open-loop
    To_gain = 20 * np.log10(np.abs(To_resp)) # 単位[dB]
    To_gain_max = np.max(To_gain)
    To_gain_min = np.min(To_gain)

    ## Real and imaginary parts, open-loop
    To_real = np.real(To_resp)
    To_imag = np.imag(To_resp)

    # Calculating closed-loop transfer function Tc(s) = G(s)/(1 + G(s) H(s))
    ## Response to s = σ + jω, closed-loop
    Tc_resp = G(Sigma + 1j * Omega) / (1 + G(Sigma + 1j * Omega) * H(Sigma + 1j * Omega))

    ## Magnitude (gain), closed-loop
    Tc_gain = 20 * np.log10(np.abs(Tc_resp)) # 単位[dB]
    Tc_gain_max = np.max(Tc_gain)
    Tc_gain_min = np.min(Tc_gain)

    ## Real and imaginary parts, closed-loop
    Tc_real = np.real(Tc_resp)
    Tc_imag = np.imag(Tc_resp)

    # Plotting the open-loop transfer function To(s)
    ax[0].set_title(fr'Open-loop transfer function $T_o(s)$, K = {K:.3f}')

    ## Contour, open-loop
    levels_o = np.arange(-40, 41, 5)
    kwargs_o = {'levels': levels_o, 'linewidths': 0.5, 'alpha': 0.7, 'colors': ['grey']}
    cont_o = ax[0].contour(Sigma, Omega, To_gain, **kwargs_o)
    cont_o.clabel(fmt = '%1.1f', fontsize = 12)

    ### Contour-filled, open-loop
    levels_f_o = np.arange(-40, 41, 2)
    kwargs_f_o = {'levels': levels_f_o, 'extend': 'both', 'cmap': 'jet'}
    cont_f_o = ax[0].contourf(Sigma, Omega, To_gain, **kwargs_f_o)
    ax[0].set_aspect('equal')

    ### Color bar, open-loop
    if frame == 0:
        divider_o = make_axes_locatable(ax[0])
        ax_cb0 = divider_o.new_horizontal(size = '4%', pad = 0.1)
        fig.add_axes(ax_cb0)
        fig.colorbar(cont_f_o, cax = ax_cb0, label='Gain [dB]', ticks = range(-200, 200, 10))

    ### Phase, as a stream line plot, open-loop
    kwargs_s_o = {'density': 2.5, 'arrowstyle': '->', 'linewidth': 0.5, 'color': 'white'}
    ax[0].streamplot(Sigma, Omega, To_real, To_imag, **kwargs_s_o)

    ax[0].set_xlabel(r'$\sigma$')
    ax[0].set_ylabel(r'$\omega$')
    ax[0].grid()

    # Plotting the closed-loop transfer function Tc(s)
    ax[1].set_title(fr'Closed-loop transfer function $T_c(s)$, K = {K:.3f}')

    ## Contour, closed-loop
    levels_c = np.arange(-40, 41, 5)
    kwargs_c = {'levels': levels_c, 'linewidths': 0.5, 'alpha': 0.7, 'colors': ['grey']}
    cont_c = ax[1].contour(Sigma, Omega, Tc_gain, **kwargs_c)
    cont_c.clabel(fmt = '%1.1f', fontsize = 12)

    ### Contour-filled, closed-loop
    levels_f_c = np.arange(-40, 41, 2)
    kwargs_f_c = {'levels': levels_f_c, 'extend': 'both', 'cmap': 'jet'}
    cont_f_c = ax[1].contourf(Sigma, Omega, Tc_gain, **kwargs_f_c)
    ax[1].set_aspect('equal')

    ### Color bar, closed-loop
    if frame == 0:
        divider_c = make_axes_locatable(ax[1])
        ax_cb1 = divider_c.new_horizontal(size = '4%', pad = 0.1)
        fig.add_axes(ax_cb1)
        fig.colorbar(cont_f_c, cax = ax_cb1, label='Gain [dB]', ticks = range(-200, 200, 10))

    ### Phase, as a stream line plot, closed-loop
    kwargs_s_c = {'density': 2.5, 'arrowstyle': '->', 'linewidth': 0.5, 'color': 'white'}
    ax[1].streamplot(Sigma, Omega, Tc_real, Tc_imag, **kwargs_s_c)

    ax[1].set_xlim(-scale, scale)
    ax[1].set_xlabel(r'$\sigma$')
    ax[1].set_ylim(-scale, scale)
    ax[1].set_ylabel(r'$\omega$')
    ax[1].grid()

    # Adjustment of horizontal spacing
    fig.subplots_adjust(wspace = 0.35)
    
    # Which frame now drawing?
    print(f'Frame: {frame + 1}/{N_FRAMES}')

# Definition and saving of the animation
ani = animation.FuncAnimation(fig, update, frames = N_FRAMES, interval = INTERVAL, repeat_delay = REPEAT_DELAY)

print('Creating and saving...')
# ani.save('transfer-function-contour.mp4', writer = 'ffmpeg', dpi = 80)
ani.save('transfer-function-contour.gif', writer = 'imagemagick')

print('Completed!')

print(f'Time elapsed: {time.time() - t0} sec')