import subprocess

import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network


def is_GPU_available() -> bool:
    try:
        subprocess.check_output(["nvidia-smi"])
        gpu_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        gpu_available = False

    return gpu_available


def int2bin(x,num_nodes):
    return format(x,'0'+str(num_nodes)+'b')

def state2bin(state):
    bin_str = ''
    for bit in state:
        bin_str += str(bit)
    
    return bin_str

def bin2state(bin_str):
    state = []
    for i in range(len(bin_str)):
        state.append(int(bin_str[i]))

    return tuple(state) 
  
# Defining a Class
class GraphVisualization:
   
    def __init__(self):
          
        # visual is a list which stores all 
        # the set of edges that constitutes a
        # graph
        self.visual = []
        self.edge_labels = dict()
          
    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b, label=None):
        #temp = [a, b]
        temp = (a,b)
        if label is not None:
            self.edge_labels[temp] = label
        self.visual.append(temp)
          
    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.DiGraph()
        G.add_edges_from(self.visual,color='red')
        # pos = nx.spring_layout(G, seed=3113794652)
        #pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
        #nx.draw_networkx(G,pos=pos,arrows=True,edge_color='red')
        #nx.draw_networkx_edge_labels(G, pos,self.edge_labels,font_color='blue')
        #plt.tight_layout()
        #plt.axis("off")
        #plt.show()

        #nx.set_edge_attributes(G, {(e[0], e[1]): {'label': 1} for e in G.edges(data=True)})
        D = nx.drawing.nx_agraph.to_agraph(G)
        # Modify node fillcolor and edge color.
        D.node_attr.update(color='blue', style='filled', fillcolor='yellow')
        D.edge_attr.update(color='blue', arrowsize=1)
        pos = D.layout('dot')
        #D.draw(os.path.join(PATH,'graph.png'))
