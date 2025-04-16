import numpy as np# Parameters

n = 1000  # Number of nodes
m = 10  # Each new node connects to m existing nodes in Barabási-Albert graph
T = 1000  # Number of time steps for the simulation

states = ['S', 'P', 'N', 'C']
states_initial_fraction = [0.9, 0.05, 0.03, 0.02]

fp = 0.2
pos_psi = np.log(1/fp)/(T)
psi1, psi2, psi3, psi4, psi5, psi6, psi7, psi8 = pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi, pos_psi

u = 0.05
t0 = 0

sigma = 0.7
phi = 0.7
xi = 0.7

c1 = 0.1
c2 = 0.1
c3 = 0.1



def v1(t):
    return c1
def v2(t):
    return c2
def v3(t):
    return c3


confpd_alpha = 0.5
confpd_beta = 5