import numpy as np

from ..graph import create_BA_graph, assign_states, Graph
from ..parameters import states, n, m, states_initial_fraction, T, u
from ..probabilities import define_tdp, define_trans_prob
from ..counts import initialize_counts
from ..simulation import simulate_fp_diffusion
from ..visualization import visualize_counts
from ..smoothing import smoothrd
G_main = create_BA_graph(n,m)
node_states_main = assign_states(G_main, states, states_initial_fraction)
fps = np.arange(0, 1, 0.1)

rcounts = simulate_fp_diffusion(G_main, node_states_main, T, u, m, fps, save_folder='Plots/fpsl/')









