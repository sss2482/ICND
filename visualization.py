import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def visualize_probabilities(probs, start_time, end_time, num_points):
    time_range = np.linspace(start_time, end_time, num_points)
    # Compute function values over time
    
    alpha_vals = probs.alpha(time_range)
    beta_vals = probs.beta(time_range)
    gamma_vals = probs.gamma(time_range)
    delta_vals = probs.delta(time_range)
    lmbda_vals = probs.lmbda(time_range)
    epsilon_vals = probs.epsilon(time_range)
    rho_vals = probs.rho(time_range)
    mu_vals = probs.mu(time_range)

    # Plot each function
    plt.figure(figsize=(8, 6))

    plt.plot(time_range, alpha_vals, label=r'$\alpha(t)$', color='blue')
    plt.plot(time_range, beta_vals, label=r'$\beta(t)$', color='green')
    plt.plot(time_range, gamma_vals, label=r'$\gamma(t)$', color='red')
    plt.plot(time_range, delta_vals, label=r'$\delta(t)$', color='orange')
    plt.plot(time_range, lmbda_vals, label=r'$\lambda(t)$', color='purple')
    plt.plot(time_range, epsilon_vals, label=r'$\epsilon(t)$', color='cyan')
    plt.plot(time_range, rho_vals, label=r'$\rho(t)$', color='magenta')
    plt.plot(time_range, mu_vals, label=r'$\mu(t)$', color='brown')

    # Add labels and legends
    plt.title('Time Evolution of Functions')
    plt.xlabel('Time (t)')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def visualize_e(start_time, end_time, num_points, psi):
    time_range = np.linspace(start_time-t0, end_time-t0, num_points)
    pp = psi1 * time_range
    pn = (-psi1) * time_range
    # Compute function values over time
    ep_vals = np.exp(pp)
    en_vals = np.exp(pn)
    plt.plot(ep_vals)
    plt.plot(en_vals)
    plt.title('Time Evolution of Functions')
    plt.xlabel('Time (t)')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualization of graph states
def visualize_graph(G, node_states, t):
    color_map = []
    for node in G:
        if node_states[node] == 'S':
            color_map.append('gray')
        elif node_states[node] == 'P':
            color_map.append('green')
        elif node_states[node] == 'N':
            color_map.append('red')
        elif node_states[node] == 'C':
            color_map.append('orange')

    plt.figure(figsize=(10, 7))
    nx.draw(G, node_color=color_map, with_labels=False, node_size=50)
    plt.title(f"State of the Network at Time {t}")
    plt.show()

def visualize_counts(counts, title=None, save_name='figure'):
    plt.figure()  # Creates a new figure for each plot
    plt.plot(counts['S'], label='S')
    plt.plot(counts['P'], label='P')
    plt.plot(counts['N'], label='N')
    plt.plot(counts['C'], label='C')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(f'{save_name}.png')
    plt.show()