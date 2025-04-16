import random

from .graph import Graph, Network
from .counts import initialize_counts
from .probabilities import define_tdp, define_trans_prob
from .visualization import visualize_counts
from numpy.random import choice
from operator import itemgetter


# defining simulation function
def simulate_competitive_diffusion(G, node_states, counts ,T, u, trans_prob, m):
    n = len(G.G.nodes())
    new_active_nodes = set()
    for node in G.G.nodes():
        if node_states[node] == 'P' or node_states[node] == 'N' or node_states[node] == 'C':
          new_active_nodes.add(node)
    
    remove_probabilities = {
        'S' : u,
        'P' : u,
        'N' : u,
        'C' : u,
    }

    t = 0
    while (len(new_active_nodes)>0 and t<T):

        next_active_nodes = set()
        # st = counts['S'][-1]/n
        # pt = counts['N'][-1]/n
        # nt = counts['P'][-1]/n
        # ct = counts['C'][-1]/n
        
        nodes_removed = []
        all_nodes = list(G.G.nodes())
        for node in all_nodes:
          if random.random() < remove_probabilities[node_states[node]]:
            nodes_removed.append(node)
            del node_states[node]
            if node in new_active_nodes:
              new_active_nodes.remove(node)
            G.G.remove_node(node)
        for nr in nodes_removed:
            node = G.add_new_node(nr, m)
            node_states[node] = 'S'


        for node in new_active_nodes:
            neighbors = list(G.G.neighbors(node))
            state = node_states[node]
            # print('s ',state)
            for neighbor in neighbors:
                if state != node_states[neighbor]:
                    if state == 'C' and node_states[neighbor] == 'S':
                        continue
                    # print('ns ',node_states[neighbor])
                    if random.random() < trans_prob[node_states[neighbor]][state](t):
                        node_states[neighbor] = state
                        next_active_nodes.add(neighbor)

        new_active_nodes = next_active_nodes
        counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
        counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
        counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
        counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
        # print(t)
        t += 1
    return (t, node_states, counts, G)


def simulate_repeatitive_diffusion(G_main, node_states_main, itr, T, u, trans_prob, m):
    rcounts = []
    for i in range(itr):
        print(i)
        # global G
        G = G_main.copy()
        G = Graph(G)
        node_states = node_states_main.copy()
        counts = initialize_counts(node_states)
        t, node_states, counts, Gl = simulate_competitive_diffusion(G, node_states, counts, T, u, trans_prob, m)
        rcounts.append(counts)
    return rcounts


def simulate_fp_diffusion(G_main, node_states_main, T, u, m, fps, save_folder='Plots/fps/'):
   for fp in fps:
    tdp = define_tdp(fp, T)
    trans_prob = define_trans_prob(tdp)
    G = G_main.copy()
    G = Graph(G)
    node_states = node_states_main.copy()
    counts = initialize_counts(node_states)

    t, node_states, counts, Gl = simulate_competitive_diffusion(G, node_states, counts, T, u, trans_prob, m)

    visualize_counts(counts, title=f"fp_{fp}", save_name=save_folder+f"fp_{fp}")


         
   
   


def random_intervention_diffusion(G, node_states, counts ,T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi):
    n = len(G.G.nodes())
    new_active_nodes = set()
    for node in G.G.nodes():
        if node_states[node] == 'P' or node_states[node] == 'N' or node_states[node] == 'C':
          new_active_nodes.add(node)
    
    remove_probabilities = {
        'S' : u,
        'P' : u,
        'N' : u,
        'C' : u,
    }

    t = 0
    while (len(new_active_nodes)>0 and t<T):

        next_active_nodes = set()
        st = counts['S'][-1]/n
        pt = counts['N'][-1]/n
        nt = counts['P'][-1]/n
        ct = counts['C'][-1]/n
        
        nodes_removed = []
        all_nodes = list(G.G.nodes())
        for node in all_nodes:
          if random.random() < remove_probabilities[node_states[node]]:
            nodes_removed.append(node)
            del node_states[node]
            if node in new_active_nodes:
              new_active_nodes.remove(node)
            G.G.remove_node(node)
        for nr in nodes_removed:
            node = G.add_new_node(nr, m)
            node_states[node] = 'S'
        
        all_nodes = list(G.G.nodes())
        s_nodes = []
        n_nodes = []
        c_nodes = []
        for node in all_nodes:
            if node_states[node] == 'S':
               s_nodes.append(node)
            elif node_states[node] == 'N':
               n_nodes.append(node)
            elif node_states[node] == 'C':
               c_nodes.append(node)
         
        correction_nodes = choice(n_nodes, int(c1*len(n_nodes)), replace=False)
        guidance_nodes = choice(c_nodes, int(c2*len(c_nodes)), replace=False)
        prevention_nodes = choice(s_nodes, int(c3*len(s_nodes)), replace=False)

        for node in correction_nodes:
           if random.random() < phi:
              node_states[node] = 'P'
              next_active_nodes.add(node)
        
        for node in guidance_nodes:
           if random.random() < sigma:
              node_states[node] = 'P'
              next_active_nodes.add(node)
        
        for node in prevention_nodes:
           if random.random() < xi:
              node_states[node] = 'P'
              new_active_nodes.add(node)

        for node in new_active_nodes:
            neighbors = list(G.G.neighbors(node))
            state = node_states[node]
            # print('s ',state)
            for neighbor in neighbors:
                if state != node_states[neighbor]:
                    if state == 'C' and node_states[neighbor] == 'S':
                        continue
                    # print('ns ',node_states[neighbor])
                    if random.random() < trans_prob[node_states[neighbor]][state](t):
                        node_states[neighbor] = state
                        next_active_nodes.add(neighbor)

        new_active_nodes = next_active_nodes
        counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
        counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
        counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
        counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
        # print(t)
        t += 1
    return (t, node_states, counts, G)


def  degree_intervention_diffusion(G, node_states, counts ,T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi):
    n = len(G.G.nodes())
    new_active_nodes = set()
    for node in G.G.nodes():
        if node_states[node] == 'P' or node_states[node] == 'N' or node_states[node] == 'C':
          new_active_nodes.add(node)
    
    remove_probabilities = {
        'S' : u,
        'P' : u,
        'N' : u,
        'C' : u,
    }

    t = 0
    while (len(new_active_nodes)>0 and t<T):

        next_active_nodes = set()   
        
        for node in new_active_nodes:
            neighbors = list(G.G.neighbors(node))
            state = node_states[node]
            # print('s ',state)
            for neighbor in neighbors:
                if state != node_states[neighbor]:
                    if state == 'C' and node_states[neighbor] == 'S':
                        continue
                    # print('ns ',node_states[neighbor])
                    if random.random() < trans_prob[node_states[neighbor]][state](t):
                        node_states[neighbor] = state
                        next_active_nodes.add(neighbor)

        nodes_removed = []
        all_nodes = list(G.G.nodes())
        for node in all_nodes:
          if random.random() < remove_probabilities[node_states[node]]:
            nodes_removed.append(node)
            del node_states[node]
            # if node in new_active_nodes:
            #   new_active_nodes.remove(node)
            G.G.remove_node(node)
        for nr in nodes_removed:
            node = G.add_new_node(nr, m)
            node_states[node] = 'S'
        
        all_nodes = list(G.G.nodes())
        s_nodes = []
        n_nodes = []
        c_nodes = []
        for node in all_nodes:
            if node_states[node] == 'S':
               s_nodes.append([node, G.G.degree[node]])
            elif node_states[node] == 'N':
               n_nodes.append([node, G.G.degree[node]])
            elif node_states[node] == 'C':
               c_nodes.append([node, G.G.degree[node]])
        
        s_nodes.sort(reverse=True, key=lambda x: x[1])
        n_nodes.sort(reverse=True, key=lambda x: x[1])
        c_nodes.sort(reverse=True, key=lambda x: x[1])

        

        correction_nodes = [n_nodes[i][0] for i in range(int(c1*len(n_nodes)))]
        guidance_nodes = [c_nodes[i][0] for i in range(int(c2*len(c_nodes)))]
        prevention_nodes = [s_nodes[i][0] for i in range(int(c3*len(s_nodes)))]


        

        for node in correction_nodes:
           if random.random() < phi:
              node_states[node] = 'P'
              next_active_nodes.add(node)
        
        for node in guidance_nodes:
           if random.random() < sigma:
              node_states[node] = 'P'
              next_active_nodes.add(node)
        
        for node in prevention_nodes:
           if random.random() < xi:
              node_states[node] = 'P'
              next_active_nodes.add(node)

        

        new_active_nodes = next_active_nodes
        counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
        counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
        counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
        counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
        # print(t)
        t += 1

    return (t, node_states, counts, G)


def repeatitive_degree_intervention_diffusion(G_main, node_states_main, T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi, itr):
    rcounts = []
    for i in range(itr):
        print(i)
        G = G_main.copy()
        G = Graph(G)
        node_states = node_states_main.copy()
        counts = initialize_counts(node_states)
        t, node_states, counts, Gl = degree_intervention_diffusion(G, node_states, counts, T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi)
        rcounts.append(counts)
    return rcounts


class Simulation():
    def __init__(self, intervention=True):
        self.intervention = intervention
        pass
    def run(self, N, counts ,T, u, m, c1, c2, c3, sigma, phi, xi ):
        n = len(N.G.nodes())
        new_active_nodes = set()
        for node in N.G.nodes():
            if N.node_states[node] == 'P' or N.node_states[node] == 'N' or N.node_states[node] == 'C':
                new_active_nodes.add(node)
        
        remove_probabilities = {
            'S' : u,
            'P' : u,
            'N' : u,
            'C' : u,
        }

        t = 0
        while (len(new_active_nodes)>0 and t<T):

            next_active_nodes = set()   
            
            for node in new_active_nodes:
                neighbors = list(N.G.neighbors(node))
                state = N.node_states[node]
                # print('s ',state)
                for neighbor in neighbors:
                    if state != N.node_states[neighbor]:
                        if state == 'C' and N.node_states[neighbor] == 'S':
                            continue
                        # print('ns ',node_states[neighbor])
                        new_neighbor_state = N.new_state(neighbor, node, t)
                        if new_neighbor_state != N.node_states[neighbor]:
                            N.node_states[neighbor] = new_neighbor_state
                            next_active_nodes.add(neighbor)

                        
                        # if random.random() < trans_prob[node_states[neighbor]][state](t):
                        #     node_states[neighbor] = state
                        #     next_active_nodes.add(neighbor)

            
            # N.remove_and_add_nodes_general(u,m)
            nodes_removed = []
            all_nodes = list(N.G.nodes())
            for node in all_nodes:
                if random.random() < remove_probabilities[N.node_states[node]]:
                    nodes_removed.append(node)
                    del N.node_states[node]
                    if node in next_active_nodes:
                        next_active_nodes.remove(node)
                    N.G.remove_node(node)
            for nr in nodes_removed:
                node = N.add_new_node(nr, m)
                N.node_states[node] = 'S'
            

            if self.intervention:
                all_nodes = list(N.G.nodes())
                s_nodes = []
                n_nodes = []
                c_nodes = []
                for node in all_nodes:
                    if N.node_states[node] == 'S':
                        s_nodes.append([node, N.G.degree[node]])
                    elif N.node_states[node] == 'N':
                        n_nodes.append([node, N.G.degree[node]])
                    elif N.node_states[node] == 'C':
                        c_nodes.append([node, N.G.degree[node]])
                
                s_nodes.sort(reverse=True, key=lambda x: x[1])
                n_nodes.sort(reverse=True, key=lambda x: x[1])
                c_nodes.sort(reverse=True, key=lambda x: x[1])

                

                correction_nodes = [n_nodes[i][0] for i in range(int(c1*len(n_nodes)))]
                guidance_nodes = [c_nodes[i][0] for i in range(int(c2*len(c_nodes)))]
                prevention_nodes = [s_nodes[i][0] for i in range(int(c3*len(s_nodes)))]


                

                for node in correction_nodes:
                    if random.random() < phi:
                        N.node_states[node] = 'P'
                        next_active_nodes.add(node)
                
                for node in guidance_nodes:
                    if random.random() < sigma:
                        N.node_states[node] = 'P'
                        next_active_nodes.add(node)
                
                for node in prevention_nodes:
                    if random.random() < xi:
                        N.node_states[node] = 'P'
                        next_active_nodes.add(node)

            

            new_active_nodes = next_active_nodes
            counts['S'].append(0)
            counts['P'].append(0)
            counts['N'].append(0)
            counts['C'].append(0)
            for state in N.node_states.values():
                counts[state][-1] += 1
            # counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
            # counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
            # counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
            # counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
            # print(t)
            t += 1

        return (t, counts, N)
    
    def repeatitive_run(self, n_iter, G_main, node_states_main, trans_prob, confpd_alpha, confpd_beta, T, u, m, c1, c2, c3, sigma, phi, xi):
        rcounts = []
        for i in range(n_iter):
            print(i)
            G = G_main.copy()
            node_states = node_states_main.copy()
            N = Network(G, node_states, trans_prob, confpd_alpha, confpd_beta)
            counts = initialize_counts(node_states)

            t, counts, N = self.run(N, counts, T, u, m, c1, c2, c3, sigma, phi, xi)

            # t, node_states, counts, Gl = degree_intervention_diffusion(G, node_states, counts, T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi)
            rcounts.append(counts)
        return rcounts
    
    def fp_run(self, G_main, node_states_main, trans_prob, confpd_alpha, confpd_beta, T, u, m, c1, c2, c3, sigma, phi, xi, fps, save_folder='Plots/comb/fps/'):
        for fp in fps:
            tdp = define_tdp(fp, T)
            trans_prob = define_trans_prob(tdp)
            G = G_main.copy()
            node_states = node_states_main.copy()
            N = Network(G, node_states, trans_prob, confpd_alpha, confpd_beta)
            counts = initialize_counts(node_states)
            t, counts, N = self.run(N, counts, T, u, m, c1, c2, c3, sigma, phi, xi)

            # t, node_states, counts, Gl = simulate_competitive_diffusion(G, node_states, counts, T, u, trans_prob, m)

            visualize_counts(counts, title=f"fp_{fp}", save_name=save_folder+f"fp_{fp}")

    
    


def run_sp(self, G, node_states, counts ,T, u, trans_prob, m, c1, c2, c3, sigma, phi, xi ):
        n = len(G.G.nodes())
        new_active_nodes = set()
        for node in G.G.nodes():
            if node_states[node] == 'P' or node_states[node] == 'N' or node_states[node] == 'C':
                new_active_nodes.add(node)
        
        remove_probabilities = {
            'S' : u,
            'P' : u,
            'N' : u,
            'C' : u,
        }

        t = 0
        while (len(new_active_nodes)>0 and t<T):

            next_active_nodes = set()   
            
            for node in new_active_nodes:
                neighbors = list(G.G.neighbors(node))
                state = node_states[node]
                # print('s ',state)
                for neighbor in neighbors:
                    if state != node_states[neighbor]:
                        if state == 'C' and node_states[neighbor] == 'S':
                            continue
                        # print('ns ',node_states[neighbor])
                        if random.random() < trans_prob[node_states[neighbor]][state](t):
                            node_states[neighbor] = state
                            next_active_nodes.add(neighbor)

            nodes_removed = []
            all_nodes = list(G.G.nodes())
            for node in all_nodes:
                if random.random() < remove_probabilities[node_states[node]]:
                    nodes_removed.append(node)
                    del node_states[node]
                    # if node in new_active_nodes:
                    #   new_active_nodes.remove(node)
                    G.G.remove_node(node)
            for nr in nodes_removed:
                node = G.add_new_node(nr, m)
                node_states[node] = 'S'
            
            all_nodes = list(G.G.nodes())
            s_nodes = []
            n_nodes = []
            c_nodes = []
            for node in all_nodes:
                if node_states[node] == 'S':
                    s_nodes.append([node, G.G.degree[node]])
                elif node_states[node] == 'N':
                    n_nodes.append([node, G.G.degree[node]])
                elif node_states[node] == 'C':
                    c_nodes.append([node, G.G.degree[node]])
            
            s_nodes.sort(reverse=True, key=lambda x: x[1])
            n_nodes.sort(reverse=True, key=lambda x: x[1])
            c_nodes.sort(reverse=True, key=lambda x: x[1])

            

            correction_nodes = [n_nodes[i][0] for i in range(int(c1*len(n_nodes)))]
            guidance_nodes = [c_nodes[i][0] for i in range(int(c2*len(c_nodes)))]
            prevention_nodes = [s_nodes[i][0] for i in range(int(c3*len(s_nodes)))]


            

            for node in correction_nodes:
                if random.random() < phi:
                    node_states[node] = 'P'
                    next_active_nodes.add(node)
            
            for node in guidance_nodes:
                if random.random() < sigma:
                    node_states[node] = 'P'
                    next_active_nodes.add(node)
            
            for node in prevention_nodes:
                if random.random() < xi:
                    node_states[node] = 'P'
                    next_active_nodes.add(node)

            

            new_active_nodes = next_active_nodes
            counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
            counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
            counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
            counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
            # print(t)
            t += 1

        return (t, node_states, counts, G)


# defining simulation function
def simulate_two_diffusion(G, node_states, counts ,T, u, trans_prob, m):
    n = len(G.G.nodes())
    new_active_nodes = set()
    for node in G.G.nodes():
        if node_states[node] == 'P' or node_states[node] == 'N' or node_states[node] == 'C':
          new_active_nodes.add(node)
    
    remove_probabilities = {
        'S' : u,
        'P' : u,
        'N' : u,
        'C' : u,
    }

    t = 0
    while (len(new_active_nodes)>0 and t<T):

        next_active_nodes = set()
        # st = counts['S'][-1]/n
        # pt = counts['N'][-1]/n
        # nt = counts['P'][-1]/n
        # ct = counts['C'][-1]/n
        
        nodes_removed = []
        all_nodes = list(G.G.nodes())
        for node in all_nodes:
          if random.random() < remove_probabilities[node_states[node]]:
            nodes_removed.append(node)
            del node_states[node]
            if node in new_active_nodes:
              new_active_nodes.remove(node)
            G.G.remove_node(node)
        for nr in nodes_removed:
            node = G.add_new_node(nr, m)
            node_states[node] = 'S'


        for node in new_active_nodes:
            neighbors = list(G.G.neighbors(node))
            state = node_states[node]
            # print('s ',state)
            for neighbor in neighbors:
                if state != node_states[neighbor]:
                    if state == 'C' and node_states[neighbor] == 'S':
                        continue
                    # print('ns ',node_states[neighbor])
                    trust = G.edges[node, neighbor]['trust']
                    bias  = G.nodes[neighbor]['bias']
                    mul_bias = 1
                    if state == 'P':
                       mul_bias = 1-bias
                       G.nodes[neighbor]['P_freshness'] = t
                    if state == 'N': 
                       mul_bias = bias
                       G.nodes[neighbor]['N_freshness'] = t
                    if state == 'C':
                       G.nodes[neighbor]['P_freshness'] = t
                       G.nodes[neighbor]['N_freshness'] = t
                    
                    
                    
                    if random.random() < trans_prob[node_states[neighbor]][state](t) * trust * mul_bias :
                        node_states[neighbor] = state
                        next_active_nodes.add(neighbor)
                    

        new_active_nodes = next_active_nodes
        counts['S'].append(sum(1 for state in node_states.values() if state == 'S'))
        counts['P'].append(sum(1 for state in node_states.values() if state == 'P'))
        counts['N'].append(sum(1 for state in node_states.values() if state == 'N'))
        counts['C'].append(sum(1 for state in node_states.values() if state == 'C'))
        # print(t)
        t += 1
    return (t, node_states, counts, G)

      
   


