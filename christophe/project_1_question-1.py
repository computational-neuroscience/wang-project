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
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint as binofit

sns.set(font_scale=1.3)
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
sigma = 0.02    # nA
#%% Putting everything together
mu_1 = 30 # Mean input for neuron 1 [nA]
mu_2 = 30 # Mean input for neuron 2 [nA]

t_1 = 500  # Start stimulus presentation [ms]
t_2 = 1500 # Stop stimulus presentation  ]ms]

total_steps = 3000 # Total simulation time [ms]

def run_simulation(params={'mu_1':30,'mu_2':30}):
    results = np.zeros((4,total_steps))
    
    I_b1 = 0
    I_b2 = 0
    
    s_1 = 0.1
    s_2 = 0.1
    for t in range(total_steps):
        if t_1 < t < t_2:
            mu_1 = params['mu_1']
            mu_2 = params['mu_2']
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
    return results
    
results = run_simulation({'mu_1':60,'mu_2':00})
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
#%% Coin-toss experiment
n_runs = 500
pop_1_wins = np.zeros(n_runs)

for i in range(500):
    results = run_simulation()
    if results[0,1500] > results[1,1500]:
        pop_1_wins[i] = 1
                  
plt.plot(np.cumsum(pop_1_wins)/np.arange(1,n_runs+1))
plt.axhline(0.5,color='k',linestyle='--')
plt.xlabel('# Simulations')
plt.title('Proportion of population 1 wins')
#%% Stimulus coherence values
mu_0 = 30
c_prime = [0.032, 0.064, 0.128, 0.256, 0.512, 0.85, 1.0]

mu = np.zeros([2,len(c_prime)])
for i, c in enumerate(c_prime):
    mu_1 = mu_0*(1 + c)
    mu_2 = mu_0*(1 - c)
    mu[0,i] = mu_1
    mu[1,i] = mu_2
plt.plot(np.log(c_prime),mu[0,:])
plt.plot(np.log(c_prime),mu[1,:])
plt.legend([r'$\mu_1$',r'$\mu_2$'])
plt.xlabel('coherence')
plt.title('Input as a function of coherence')

#%% Simulate with different coherence values
n_runs = 200
wins = np.zeros(len(c_prime))

for i, c in enumerate(c_prime):
    mu_1 = mu_0*(1 + c)
    mu_2 = mu_0*(1 - c)
    
    for j in range(n_runs):
        sim_result = run_simulation({'mu_1': mu_1, 'mu_2' : mu_2})
        if sim_result[0,1500] > sim_result[1,1500]:
            wins[i] += 1

y = wins/n_runs
err = binofit(wins,n_runs,0.05)
plt.title('Psychometric curve')
plt.errorbar(np.log(c_prime),y,yerr = [y-err[0],err[1]-y],fmt='.-')
plt.xlabel(r"$\log{ (c')}$")
plt.ylabel('P(correct)')
plt.ylim(0.5,1.05)

#%% Reaction time task
n_runs = 200
example_curves = np.zeros([2,len(c_prime), total_steps])
rt = np.zeros([2,len(c_prime),n_runs])

y_1 = np.zeros(len(c_prime))
y_2 = np.zeros(len(c_prime))
y_3 = np.zeros(len(c_prime))
y_4 = np.zeros(len(c_prime))

threshold = 15
for i,c in enumerate (c_prime):
    mu_1 = mu_0*(1 + c)
    mu_2 = mu_0*(1 - c)
    
    wins = 0
    for j in range(n_runs):
        sim_result = run_simulation({'mu_1': mu_1, 'mu_2' : mu_2})
        
        if len(np.where(sim_result[0,:]>threshold))>0:
            threshold_time_1 = np.argmax(sim_result[0,:] > threshold)
        else:
            threshold_time_1 = total_steps
            
        if len(np.where(sim_result[1,:]>threshold)[0])>0:
            threshold_time_2 = np.argmax(sim_result[1,:] > threshold)
        else:
            threshold_time_2 = total_steps
            
        if threshold_time_1 < threshold_time_2:
            rt[0,i,j] = 0
            rt[1,i,j] = threshold_time_1
        else:
            rt[0,i,j] = 1
            rt[1,i,j] = threshold_time_2     
              
    y_1[i] = np.mean(rt[1,i, rt[0,i,:]==0])
    y_2[i] = np.mean(rt[1,i, rt[0,i,:]==1])
    y_3[i] = np.mean(rt[1,i,:])
    y_4[i] = len(np.where(rt[0,i,:] == 0)[0])/float(n_runs)
    
plt.plot(np.log(c_prime), y_1)
plt.plot(np.log(c_prime), y_2)
plt.plot(np.log(c_prime), y_3,'--k')
plt.legend(['Population #1','Population #2'])

plt.plot(np.log(c_prime),y_4)

plt.plot(sim_result[0,:])
plt.plot(sim_result[1,:])
plt.axhline(15)