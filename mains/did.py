from ..graph import create_BA_graph, assign_node_edge_parameters, assign_states, Graph, Network
from ..parameters import states, n, m, states_initial_fraction, T, u, phi, sigma, xi, c1, c2, c3, confpd_alpha, confpd_beta
from ..probabilities import define_tdp, define_trans_prob
from ..counts import initialize_counts
from ..simulation import degree_intervention_diffusion, Simulation
from ..visualization import visualize_counts
from ..smoothing import smoothrd


G_main = create_BA_graph(n,m)
G_main = assign_node_edge_parameters(G_main)
node_states_main = assign_states(G_main, states, states_initial_fraction)


fp = 0.08
tdp = define_tdp(fp, T)
trans_prob = define_trans_prob(tdp)

G = G_main.copy()
# G = Graph(G)


node_states = node_states_main.copy()

N = Network(G, node_states, trans_prob, confpd_alpha, confpd_beta)

counts = initialize_counts(node_states)

sim = Simulation(intervention=False)
t, counts, N = sim.run(N, counts ,T, u, m, c1, c2, c3, sigma, phi, xi)

# t, node_states, counts, G = degree_intervention_diffusion(G, node_states, counts, T, u, trans_prob, m, 0.1, 0.1, 0.1, sigma, phi, xi)
N.plot_bias_distribution()
visualize_counts(counts, save_name='Plots/inplots/ndid/fp_28_inF')


# smcounts = smoothrd(rcounts, T)

# visualize_counts(smcounts, save_name='Plots/smplots/smcountssample3')





