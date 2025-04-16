#libraries
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
from math import tanh
from scipy.stats import beta
import seaborn as sns

from .helper import ProbDistribution

def create_BA_graph(n, m):
    # Create Barabási-Albert graph
    G_main = nx.barabasi_albert_graph(n, m)
    # assigning states to nodes

    return G_main

def assign_node_edge_parameters(G):
    for edge in G.edges():
        G.edges[edge]['trust'] = random.betavariate(alpha=21, beta=9)
    
    for node in G.nodes():
        G.nodes[node]['bias'] = random.betavariate(alpha=1, beta=1)
        G.nodes[node]['timeN'] = 0
        G.nodes[node]['timeP'] = 0
    return G

def assign_states(G_main, states, states_initial_fraction):
    nodes = choice(states,  len(G_main.nodes()),
                p=states_initial_fraction)
    i = 0
    node_states_main = {}
    for node in G_main.nodes():
        # G.nodes[node]['state'] = draw[i]
        node_states_main[node] = nodes[i]
        i += 1
    return node_states_main

#static
class Graph():

    def __init__(self, G):
        self.G = G

    # add function for graph
    def add_new_node(self, r, m):
        """
        Add a new node with 'm' edges to existing nodes,
        automatically connecting based on the Barabási-Albert preferential attachment.
        """
        new_node = r  # Next available node index
        self.G.add_node(new_node)

        # Attach the new node to 'm' existing nodes preferentially by degree
        targets = set()
        degrees = dict(self.G.degree())
        total_degree = sum(degrees.values())
        node_selection_probabilities = [degrees[node]/total_degree for node in degrees.keys()]
        nodes = choice(list(degrees.keys()), m, replace=False, p=node_selection_probabilities)
        # while len(targets) < m:
        #     node = random.choices(list(degrees.keys()), weights=degrees.values(), k=1)[0]
        #     if node not in targets:  # Prevent duplicate edges
        #         targets.add(node)
        for node in nodes:
            targets.add(node)

        # Add edges between the new node and the selected target nodes
        for target in targets:
            self.G.add_edge(new_node, target)
            self.G.edges[(new_node, target)]['trust'] = random.betavariate(21,9)
        
        self.G.nodes[new_node]['bias'] = random.betavariate(1,1)
        self.G.nodes[new_node]['timeN'] = 0
        self.G.nodes[new_node]['timeP'] = 0
        return new_node
    

class Network(Graph):
    
    def __init__(self, G, node_states, trans_prob, confpd_alpha, confpd_beta):
        super().__init__(G)
        self.node_states = node_states
        self.trans_prob = trans_prob
        self.confpd = ProbDistribution(alpha=confpd_alpha, beta=confpd_beta)
    
    def add_new_node(self, r, m):
        return super().add_new_node(r, m)
    
    def update_bias(self, bias, changex):
        x = np.arctanh(2 * bias - 1)
        x += changex

        bias = (1+tanh(x))/2
        return bias

    def new_bias_general(self, bias, state_b):
        dir = 1 if state_b == 'N' else -1

        changex = dir * 0.05
        
        return self.update_bias(bias, changex)
    
    def new_bias_accepting(self, bias, state_b):
        dir = 1 if state_b == 'N' else -1

        changex = dir * 0.15
        return self.update_bias(bias, changex)
    
    def new_bias_general_confusion(self, bias):
        dir = 1 if bias <0 else -1

        changex = dir * 0.05
        return self.update_bias(bias, changex)
    
    def new_bias_accepting_confusion(self, bias):
        dir = 1 if bias <0 else -1

        changex = dir * 0.15
        return self.update_bias(bias, changex)

    def new_state(self, a, b, t):
        trust = self.G.edges[(a,b)]['trust']
        bias = self.G.nodes[a]['bias']
        state_a = self.node_states[a]
        state_b = self.node_states[b]
        if state_a == 'S':
            if state_b == 'P':
                bias = 1 - bias
                self.G.nodes[a]['timeP'] = t
            else:
                self.G.nodes[a]['timeN'] = t
            prob = self.trans_prob[state_a][state_b](t) * trust * bias

            if random.random() < prob:
                self.G.nodes[a]['bias'] = self.new_bias_accepting(self.G.nodes[a]['bias'], state_b)
                return state_b
            self.G.nodes[a]['bias'] = self.new_bias_general(self.G.nodes[a]['bias'], state_b)
            return state_a
        
        
        elif state_a == 'P':
            if state_b == 'N':
                prob = self.trans_prob[state_a][state_b](t) * trust * bias
                freshnessN = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeN']))])
                freshnessP = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeP']))])
                probN = min([prob, prob * (1+tanh(freshnessN - freshnessP))])
                probC = self.confpd.prob_val(abs(bias - 0.5)) * self.confpd.prob_val(abs(trust - (1 - bias)))
                x = 0.1
                probP = x
                
                self.G.nodes[a]['timeN'] = t
                
                sumP = probN + probC + probP
                probfN = probN/sumP
                probfC = probC/sumP
                probfP = probP/sumP


                proba = random.random()

                if proba<probfN:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting(bias, state_b)
                    return 'N'
                elif proba<probfN+probfC:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting_confusion(bias)
                    return 'C'
                else:
                    self.G.nodes[a]['bias'] = self.new_bias_general(bias, state_b)
                    return state_a
            if state_b == 'C':
                prob = self.trans_prob[state_a][state_b](t) * trust * bias
                
                self.G.nodes[a]['timeN'] = t
                self.G.nodes[a]['timeP'] = t

                proba = random.random()

                if proba<prob:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting_confusion(bias)
                    return 'C'
                
                self.G.nodes[a]['bias'] = self.new_bias_general_confusion(bias)
                return state_a
        
        elif state_a == 'N':

            if state_b == 'P':
                prob = self.trans_prob[state_a][state_b](t) * trust * (1-bias)
                freshnessN = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeN']))])
                freshnessP = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeP']))])
                probP = min([prob, prob * (1+tanh(freshnessP - freshnessN))])
                probC = self.confpd.prob_val(abs(bias - 0.5)) * self.confpd.prob_val(abs(trust - bias))
                x = 0.1
                probN = x

                self.G.nodes[a]['timeP'] = t

                sumProb = probN + probC + probP
                probfP = probP/sumProb
                probfC = probC/sumProb
                probfN = probN/sumProb

                proba = random.random()

                if proba<probfP:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting(bias, state_b)
                    return 'P'
                elif proba<probfP+probfC:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting_confusion(bias)
                    return 'C'
                else:
                    self.G.nodes[a]['bias'] = self.new_bias_general(bias, state_b)
                    return state_a
                

            if state_b == 'C':
                prob = self.trans_prob[state_a][state_b](t) * trust * (1-bias)
                
                self.G.nodes[a]['timeN'] = t
                self.G.nodes[a]['timeP'] = t

                proba = random.random()

                if proba<prob:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting_confusion(bias)
                    return 'C'
                self.G.nodes[a]['bias'] = self.new_bias_general(bias, state_b)
                return state_a
            
        else:
            if state_b == 'P':
                prob = self.trans_prob[state_a][state_b](t) * trust * (1-bias)
                freshnessN = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeN']))])
                freshnessP = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeP']))])
                prob = min([prob, prob * (1+tanh(freshnessP - freshnessN))])
                
                self.G.nodes[a]['timeP'] = t

                proba  = random.random()

                if proba< prob:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting(bias, state_b)
                    return 'P'
                
                self.G.nodes[a]['bias'] = self.new_bias_general(bias, state_b)
                return state_a
            if state_b == 'N':
                prob = self.trans_prob[state_a][state_b](t) * trust * bias
                freshnessN = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeN']))])
                freshnessP = max([0,1 - (0.01 * (t - self.G.nodes[a]['timeP']))])
                prob = min([prob, prob * (1+tanh(freshnessN - freshnessP))])

                self.G.nodes[a]['timeN'] = t

                proba  = random.random()

                if proba < prob:
                    self.G.nodes[a]['bias'] = self.new_bias_accepting(bias, state_b)
                    return 'N'
                self.G.nodes[a]['bias'] = self.new_bias_general(bias, state_b)
                return state_a
        

    def change_state(self, a, b, t):
        self.node_states[a] = self.new_state(a, b, t)

    
    def remove_and_add_nodes_general(self, u, m):
        nodes_removed = []
        all_nodes = list(self.G.nodes())
        for node in all_nodes:
            if random.random() < u:
                nodes_removed.append(node)
                del self.node_states[node]
                # if node in new_active_nodes:
                #   new_active_nodes.remove(node)
                self.G.remove_node(node)
        for nr in nodes_removed:
            node = self.add_new_node(nr, m)
            self.node_states[node] = 'S'

    
    def bias_list(self):
        biases = []
        for node in self.G.nodes:
            biases.append(self.G.nodes[node]['bias'])
        return biases
    
    def plot_bias_distribution(self):
        bias_list= self.bias_list()
        kde = True
        bins = 10
        # Example list of biases
       

        # Function to plot the distribution
        # def plot_bias_distribution(bias_list, bins=10, kde=True):
        plt.figure(figsize=(8, 6))
        
        # Using seaborn for a better plot
        sns.histplot(bias_list, bins=bins, kde=kde, color='skyblue', alpha=0.7, edgecolor='black')
        
        plt.title("Distribution of Bias", fontsize=16)
        plt.xlabel("Bias", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(axis='y', alpha=0.75)
        plt.show()

        # Call the function with the bias list
        # plot_bias_distribution(biases)

    
    
    


            

                
                


                
                
