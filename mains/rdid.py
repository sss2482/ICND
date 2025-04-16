from ..graph import create_BA_graph, assign_states, Graph
from ..parameters import states, n, m, states_initial_fraction, T, u, c1, c2, c3, phi, sigma, xi
from ..probabilities import define_tdp, define_trans_prob
from ..counts import initialize_counts
from ..simulation import repeatitive_degree_intervention_diffusion
from ..visualization import visualize_counts
from ..smoothing import smoothrd
G_main = create_BA_graph(n,m)
node_states_main = assign_states(G_main, states, states_initial_fraction)

fp = 0.28
tdp = define_tdp(fp, T)
trans_prob = define_trans_prob(tdp)
itr = 2000
rcounts = repeatitive_degree_intervention_diffusion(G_main, node_states_main, T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi, itr)

smcounts = smoothrd(rcounts, T)

visualize_counts(smcounts, save_name='Plots/inplots/rdid/fp_028')





