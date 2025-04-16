from ..graph import create_BA_graph, assign_states, Graph
from ..parameters import states, n, m, states_initial_fraction, T, u, phi, sigma, xi, c1, c2, c3
from ..probabilities import define_tdp, define_trans_prob
from ..counts import initialize_counts
from ..simulation import simulate_competitive_diffusion
from ..visualization import visualize_counts
from ..smoothing import smoothrd


G_main = create_BA_graph(n,m)

node_states_main = assign_states(G_main, states, states_initial_fraction)

fp = 0.7
tdp = define_tdp(fp, T)
trans_prob = define_trans_prob(tdp)

G = G_main.copy()
G = Graph(G)
node_states = node_states_main.copy()
counts = initialize_counts(node_states)


t, node_states, counts, G = simulate_competitive_diffusion(G, node_states, counts ,T, u, trans_prob, m)

visualize_counts(counts, save_name='Plots/sample')

# smcounts = smoothrd(rcounts, T)

# visualize_counts(smcounts, save_name='Plots/smplots/smcountssample3')





