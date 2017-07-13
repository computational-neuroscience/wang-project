# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:02:22 2017

@author: Christophe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:33:09 2017

@author: Christophe
"""
import numpy as np
import matplotlib.pyplot as plt

## Input-output curve
a = 270    # Hz/nA
b = 108    # Hz
d = 0.154  # s

def rate(I, a = 270.0, b = 108.0, d = 0.154):
    return (a*I - b)/(1 - np.exp(-d*(a*I - b)))

## Synaptic drive variables
gamma = 0.641
theta = 1.0
tau_s = 100     # ms
g_E   = 0.2609  # nA
g_I   = 0.0497  # nA
g_ext = 0.00052 # nA

## background input: Ornstein-Uhlenbeck process
I_0   = 0.3255  # nA
tau_0 = 2.0     # ms
sigma = 0.005    # nA
#%% Putting everything together
mu_1 = 30 # Mean input for neuron 1 [nA]
mu_2 = 30 # Mean input for neuron 2 [nA]

s_1 = 0.1
s_2 = 0.1

t_1 = 500  # Start stimulus presentation [ms]
t_2 = 1500 # Stop stimulus presentation  ]ms]

total_steps = 3000 # Total simulation time [ms]

I_b1 = 0#I_0
I_b2 = 0#I_0

results = np.zeros((4,total_steps))

for t in range(total_steps):
    if t_1 < t < t_2:
        mu_1 = 30
        mu_2 = 30
    else:
        mu_1 = 0
        mu_2 = 0
        
    # Update input currents and calculate predicted rate
    I_1 = g_E*s_1 - g_I*s_2 + I_b1 + g_ext*mu_1
    I_2 = g_E*s_2 - g_I*s_1 + I_b2 + g_ext*mu_2
    
    F_1 = rate(I_1)/1000.0
    F_2 = rate(I_2)/1000.0
    
    # Calculate differentials for synaptic drives
    ds_1 = theta * (F_1*gamma*(1-s_1) - s_1/tau_s)
    ds_2 = theta * (F_2*gamma*(1-s_2) - s_2/tau_s)
    
    # Calculate differentials for background noise
    dI_b1 = -(I_b1 - I_0)/tau_0 + np.random.normal()*np.sqrt(tau_0*sigma**2)
    dI_b2 = -(I_b2 - I_0)/tau_0 + np.random.normal()*np.sqrt(tau_0*sigma**2)
    
    # Update state variables
    I_b1 = I_b1 + dI_b1/tau_0
    I_b2 = I_b2 + dI_b2/tau_0
    
    s_1 = s_1 + ds_1
    s_2 = s_2 + ds_2
    
    # Save the results
    results[0,t] = F_1*1000
    results[1,t] = F_2*1000
    results[2,t] = s_1
    results[3,t] = s_2
           
plt.subplot(121)
plt.plot(results[0,:])
plt.plot(results[1,:])
plt.axvline(t_1,color='r')
plt.axvline(t_2,color='r')
plt.subplot(122)
plt.plot(results[2,:])
plt.plot(results[3,:])
plt.axvline(t_1,color='r')
plt.axvline(t_2,color='r')
plt.show()

plt.subplot(121)
plt.plot(results[0,:],results[1,:])
plt.subplot(122)
plt.plot(results[2,:],results[3,:])