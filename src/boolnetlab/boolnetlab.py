from typing import TypeAlias, TypeVar, Type

import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from absl.logging import flush
from markdown_it.common.html_blocks import block_names
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# from pyvis.network import Network
import boolean as bool
from collections import deque, defaultdict
from sympy.codegen.ast import continue_
from sympy.logic.boolalg import truth_table
import pandas as pd
import seaborn as sns
from typing_extensions import deprecated

import copy
import os
import random
import sympy
import time
from functools import cache

import bang
from .utils.utils import *
from .backbone import backbone

from .parse_init_states import parse_init_states_block, parse_target_states_block

try:
    import dd.cudd as bdd
    CUDD_LOADED = True
except ImportError:
    print("""WARNING: The dd.cudd module of the dd package, which provides Cython bindings to the CUDD library in C,
         is not compiled and cannot be loaded. Please refer to https://github.com/tulip-control/dd for
         installation instructions. Falling back to dd.autoref, which wraps the pure-Python Binary
         Decision Diagrams implementation dd.bdd."""
    )
    import dd.autoref as bdd
    CUDD_LOADED = False

from dd.autoref import Function
from tqdm import tqdm
import sys

# Type definitions
BNReal = TypeVar("BNReal", bound="BN_Realisation")
State: TypeAlias = tuple[int, ...]

# Inidcates whether code correctness tests should be performed. Setting to True may decrease performance.
CHECK_CORRECTNESS = True


class BN_Realisation:
    __bool_algebra = bool.BooleanAlgebra()
    __bdd = bdd.BDD()
    __bdd_ec = bdd.BDD()
    _V_INDEX = 0
    _V_LOWLINK_INDEX = 1
    _V_ON_STACK_INDEX = 2
    _FIRST_RULE = 0
    _RECURSION_LIMIT = 100000000
    _GET_SCC = 0
    _GET_CONTROL_NODES = 1
    _CONVERTED_TRUE_IN_BOOLEAN_PY_TO_STRING = "1"
    _CONVERTED_FALSE_IN_BOOLEAN_PY_TO_STRING = "0"
    _FIRST_VARIABLE_NAME_IN_ISPL_FILE = 0


    """
    Generates a random Boolean function with specified parent nodes.

        Args:
            parents (list[str]): A list of node names representing the parent nodes.

        Returns:
            tuple[str, bool, int]: A tuple containing:
                - str: The string representation of the generated function.
                - bool: Whether the generated function leads to an input node 
                  (see the `inputNodeNames()` method).
                - int: The number of essential variables, since some parent nodes 
                  may become unessential during random generation.
    """
    @staticmethod
    def __generateRandomBooleanFunction(parents: list[str]) -> tuple[str, bool, int]:
        is_fixed = False

        # Random truth_table.
        num_of_ones = random.randint(0, 2 ** len(parents))
        minterms = random.sample(range(2 ** len(parents)), num_of_ones)

        sym = sympy.symbols(parents)
        bool_formula = sympy.logic.SOPform(sym, minterms, [])
        fun = str(bool_formula)
        if fun == 'True' or fun == 'False':
            fun = fun.upper()
            is_fixed = True
            num_ps = 0
        else:
            num_ps = len(bool_formula.atoms())

        return fun, is_fixed, num_ps


    """
    Returns a list of input nodes in the Boolean network.

    An input node is defined as a node without a Boolean function, meaning its initial value 
    remains constant (unchanged). Input nodes are typically used to specify environmental 
    conditions.

    In this implementation, every node must have an associated Boolean function. Therefore, 
    a node is considered an input node if and only if its Boolean function equals the node 
    itself—for example, for node `x_i`, the Boolean function is `x_i`.

    Note:
        Nodes with constant Boolean functions (`TRUE` or `FALSE`) are **not** considered 
        input nodes, since their initial values may still be flipped, after which they 
        remain constant.

    Returns:
        list[str]: A list of node names representing the input nodes.
    """
    def getInputNodeNames(self) -> list[str]:

        input_nodes = []

        for i, fun in enumerate(self.functions):
            # syms = fun.symbols

            # # node = TRUE / FALSE
            # # if (len(syms) == 0):
            # #    input_nodes.append(self.node_names[i])

            # # A node depends only on itself.
            # if (len(syms) == 1):
            #     if str(syms.pop()) == self.node_names[i]:
            #         input_nodes.append(self.node_names[i])

            literals = fun.get_literals()
            if (len(literals) == 1):
                if str(literals[0]) == self.node_names[i]:
                    input_nodes.append(self.node_names[i])

        return input_nodes

    
    """
    Saves the Boolean network to a file in the ISPL (Interpreted Systems Programming Language) format.  
    For details on the ISPL format, see: https://vas.doc.ic.ac.uk/software/mcmas/manual.pdf

    Args:
        filepath (str): The path (including file name) where the Boolean network will be saved. 
            If the directories in the path do not exist, they will be created.
            
        initial_condition (str, optional): An optional initial condition string to include in the 
            `InitStates` section of the ISPL file. For example: 
            `"M.v_ABL1=true or M.v_ABL1=false;"`, where `v_ABL1` is a node name.

        verbose (bool): If True (default), enables verbose mode during saving.

    Returns:
        None
    """
    def save_ispl(self, filepath: str, initial_condition: str = None, verbose: bool = True) -> None:

        if verbose:
            print(f"Exporting the Boolean network to the {filepath} file in the ISPL format ... ", end='')

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)
            if verbose:
                print(f"Created folder {folder}.")

        with open(filepath, 'w') as file:

            file.write('Agent M\n')

            file.write('\tVars:\n')
            for node_name in self.node_names:
                file.write('\t\t' + node_name + ' : boolean;\n')
            file.write('\tend Vars\n')

            file.write('\tActions = {none};\n')
            file.write('\tProtocol:\n')
            file.write('\t\tOther: {none};\n')
            file.write('\tend Protocol\n')

            file.write('\tEvolution:\n')
            for node_name, fun in zip(self.node_names, self.functions):
                if str(fun) == '1':
                    fun_str = '(' + node_name + '|~' + node_name + ')'
                elif str(fun) == '0':
                    fun_str = '(' + node_name + '&~' + node_name + ')'
                else:
                    fun_str = str(fun)
                file.write('\t\t' + node_name + '=true if ' + fun_str + '=true;\n')
                file.write('\t\t' + node_name + '=false if ' + fun_str + '=false;\n')
            file.write('\tend Evolution\n')

            file.write('end Agent\n')
            file.write('\n')

            file.write('InitStates\n')
            if initial_condition is None:
                input_node_names = self.getInputNodeNames()
                if len(input_node_names) == 0:
                    # Any value for the first node
                    file.write(f'\t\tM.{self.node_names[0]}=false or M.{self.node_names[0]}=true;\n')
                else:
                    initial_condition = ''
                    for input_node_name in input_node_names:
                        initial_condition += 'M.' + input_node_name + "=false and "
                    if len(initial_condition) > 0:
                        initial_condition = initial_condition[:-5]
                    file.write('\t\t' + initial_condition + ';\n')
            else:
                file.write(initial_condition + '\n')

            file.write('end InitStates\n')

        if verbose:
            print('done.')


    """
    Reads node names and their associated Boolean functions from a file in the ISPL format.

    Args:
        path_to_ispl_file (str): The path to the Boolean network model specification file in ISPL format.

        verbose (bool, optional): If True, prints diagnostic and progress information. 
            Defaults to False.

    Returns:
        BN_Realisation: A Boolean network object containing the nodes and Boolean functions 
            specified in the ISPL file.
    """
    @classmethod
    def load_ispl(cls: Type[BNReal], path_to_ispl_file: str, verbose: bool = False) -> BNReal:
        if not os.path.isfile(path_to_ispl_file):
            raise FileNotFoundError(path_to_ispl_file)

        BN_variables = []
        BN_functions = []

        BN_functions_dict = dict()

        with open(path_to_ispl_file, "r") as ispl_file:
            line = ispl_file.readline()
            while line:
                while line.strip() != "Vars:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Vars":
                    line = line.strip()
                    gene_name = line.split(':')[0]
                    BN_variables.append(gene_name.strip())
                    line = ispl_file.readline()

                while line.strip() != "Evolution:":
                    line = ispl_file.readline()

                line = ispl_file.readline()

                while line.strip() != "end Evolution":
                    line = ispl_file.readline()
                    line = line.strip()
                    # Extract the target node name
                    line_parts = line.split(" if ")
                    target_node = line_parts[0].split('=')[0].strip()
                    # Extract the function string
                    line = line_parts[1]
                    line = line.split("=")[0]
                    BN_functions_dict[target_node] = line.strip()
                    # Skip the dual '=false' line for the target node
                    line = ispl_file.readline()

                for bn_var in BN_variables:
                    BN_functions.append(BN_functions_dict[bn_var])

                while line.strip() != "InitStates":
                    line = ispl_file.readline()
                block_text = line
                while line.strip() != "end InitStates":
                    line = ispl_file.readline()
                    if len(line.strip()) > 1:
                        if line.strip()[0:2] != "--":
                            block_text += line

                init_states_expr = parse_init_states_block(block_text)

                while line.strip() != "TargetStates":
                    line = ispl_file.readline()
                block_text = line
                while line.strip() != "end TargetStates":
                    line = ispl_file.readline()
                    if len(line.strip()) > 1:
                        if line.strip()[0:2] != "--":
                            block_text += line

                target_states_expr = parse_target_states_block(block_text)

                break

        assert len(BN_variables) == len(BN_functions), "The number of nodes does not match the number of Boolean functions."

        print(f"Loaded a Boolean network of {len(BN_variables)} nodes.")

        return BN_Realisation(BN_variables,
                              BN_functions,
                              environmental_conditions=init_states_expr,
                              target_configuration=target_states_expr,
                              verbose = verbose)


    @classmethod
    def generate_random_bn(cls, num_nodes: int, max_parent_nodes: int, min_parent_nodes: int = 1,
                           allow_self_loops: bool = True, allow_input_nodes: bool = True, mode: str = 'asynchronous',
                           verbose: bool = False):

        assert num_nodes > 0, "Number of nodes must be at least 1!"

        if max_parent_nodes > num_nodes:
            if allow_self_loops:
                print(
                    "WARNING: Maximum number of parent nodes cannot be greater than the total number of nodes. Setting max_parent_nodes to the total number of nodes.")
                max_parent_nodes = num_nodes
            else:
                print(
                    "WARNING: Maximum number of parent nodes cannot be greater than the total number of nodes. Additionally, self loops are not allowed. Setting max_parent_nodes to one less than the total number of nodes.")
                max_parent_nodes = num_nodes - 1

        if verbose:
            print(f"Generating a random Boolean network with {num_nodes} nodes ... ", end='')

        if not allow_input_nodes:
            if min_parent_nodes == 0:
                print(
                    f"WARNING: Input nodes are not allowed since allow_input_nodes is set to False. Thus, each node must have at least one parent node. Setting min_parent_nodes to 1.")
                min_parent_nodes = 1

        assert max_parent_nodes >= min_parent_nodes

        list_of_nodes = ['x' + str(i) for i in range(num_nodes)]
        input_node_indices = set()
        bn_functions = []

        for i in range(num_nodes):
            num_parents = random.randint(min_parent_nodes, max_parent_nodes)

            if allow_self_loops:
                potential_parent_nodes = list_of_nodes
            else:
                potential_parent_nodes = copy.copy(list_of_nodes)
                potential_parent_nodes.pop(i)

            parents = sorted(random.sample(potential_parent_nodes, num_parents))

            if len(parents) == 0:

                fun = 'x' + str(i)  # This will be an input node, so its initial should remain unchanged
                input_node_indices.add(i)

            else:

                fun, is_constant, num_ps = cls.__generateRandomBooleanFunction(parents)
                # Make sure that the node truly depends on at least the specified minimum number of parent nodes
                # and, if required, make sure that there are no input nodes.
                # ToDo: INEFFICIENT implementation!!!
                incorrect_function = True
                while incorrect_function:
                    fun, is_constant, num_ps = cls.__generateRandomBooleanFunction(parents)
                    incorrect_function = ((not allow_input_nodes) and fun == 'x' + str(i)) or (
                            num_ps < min_parent_nodes)

                if fun == 'x' + str(i):
                    # We have x_i(t+1) = x_i(t), so this is an input node
                    input_node_indices.add(i)

            bn_functions.append(fun)

        if verbose:
            print("done.")
            print("=============================================================================================")
            print(f"A Boolean network with {num_nodes} nodes is generated. The Boolean functions are as follows:")
            for i, f in enumerate(bn_functions):
                print(f"{list_of_nodes[i]} = {f}")
            print(f"Number of fixed-value nodes: {len(input_node_indices)}")
            print("=============================================================================================")

        bn = BN_Realisation(list_of_nodes, bn_functions, mode = mode, verbose = verbose)

        return bn


    """
    Generates the state transition graph for the Boolean network.

    Args:
        excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
            during the construction of the state transition graph.

    Returns:
        networkx.Graph: A NetworkX graph object representing the state transition graph.
    """
    def generateStateTransitionGraph(self, excludedNodes: list[str] = []) -> nx.Graph:
        G = nx.DiGraph()

        for initial_state_int in range(2 ** self.num_nodes):
            initial_state = bin2state(int2bin(initial_state_int, self.num_nodes))

            neighbor_states = self.getNeighborStates(initial_state, excludedNodes)

            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(initial_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

        return G
    

    """
    Generates a partial state transition graph for the Boolean network consisting of states reachable
    from the provided list of initial states.

    Args:

        initial_states_bin (list[str], optional): list of strings representing initial states in the
            binary string representation. If not provided, the full state transition system will be
            computed.

        excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
            during the construction of the state transition graph.

    Returns:
        networkx.Graph: A NetworkX graph object representing the state transition graph.
    """
    def _generatePartialStateTransitionGraph(self, initial_states_bin: list[str] = None, excludedNodes: list[str] = []):

        if initial_states_bin == None:
            return self.generateStateTransitionGraph(excludedNodes)

        G = nx.DiGraph()

        states_to_process = set()
        processed_states = set()

        for initial_state_bin in initial_states_bin:
            states_to_process.add(bin2state(initial_state_bin))

        while len(states_to_process) > 0:
            source_state = states_to_process.pop()

            processed_states.add(source_state)

            neighbor_states = self.getNeighborStates(source_state, excludedNodes)

            # for s in neighbor_states:
            #     print(s)
            
            edges_aux = []
            for ns in neighbor_states:
                edges_aux.append((state2bin(source_state), state2bin(ns)))
            G.add_edges_from(edges_aux)

            states_to_process = states_to_process.union(neighbor_states.difference(processed_states))

        return G


    """
    Generates a graph of the Boolean network attractor with the given attractor state.

    Args:
        attractor_state (str): An attractor state in binary string representation of the
            attractor for which the graph is to be computed.

    Returns:
        networkx.Graph: A NetworkX graph object representing the attractor.
    """
    def generateAttractorGraph(self, attractor_state: str):
        return self._generatePartialStateTransitionGraph([attractor_state])


    """
    Makes a plot of a Boolean network attractor with the given attractor state using PyGraphviz.

    Args:
        attractor_state (str):
            Binary string representing the attractor state of the attractor for which
            the heatmap will be generated.

        filename (str): Path to the file to which the image will be saved.

        nodes_relabelling (dict[str,str], optional): A dictionary for relabelling the graph nodes.
            Keys are binary strings representing attractor states.

    Returns:
        networkx.Graph: A NetworkX graph object representing the attractor.
    """
    def plotAttractor(self, attractor_state: str, filename: str, nodes_relabelling = None):

        G = self._generatePartialStateTransitionGraph([attractor_state])

        if nodes_relabelling is not None:
           nx.relabel_nodes(G, nodes_relabelling) 

        attractor_graph = nx.drawing.nx_agraph.to_agraph(G)
        attractor_graph.layout('dot')
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # struct_graph is an instant of the PyGraphviz.AGraph class
        attractor_graph.draw(filename)


    """Compute the state transition system of an attractor.

    Args:
        attractor_state (str): Binary string representing the state of the attractor 
            for which the transition system will be generated.
        extract_max_cycle (bool, optional): If True, partitions and orders the returned 
            attractor states with the longest simple cycle first, followed by remaining 
            states. If False, states are returned in default order. Defaults to False.
        verbose (bool, optional): If True, prints diagnostic and progress information. 
            Defaults to False.

    Returns:
        tuple: A 3-element tuple containing:
            - list[str]: List of attractor states,
            - int or None: The number of states in the longest cycle if 
              `extract_max_cycle` is True, otherwise None,
            - nx.DiGraph: The state transition system graph of the attractor.
    """
    def _getAttractorStatesForHeatmap(self, attractor_state: str, extractMaxCycle: bool = False, verbose: bool = False):

        G = self._generatePartialStateTransitionGraph([attractor_state])

        if extractMaxCycle:

            num_edges = G.size()

            if num_edges < 100:

                # Find all simple cycles - uses Johnson’s algorithm, which is efficient for sparse graphs.
                cycles = list(nx.simple_cycles(G))

                # Find the longest cycle
                max_cycle = max(cycles, key=len)
                if verbose:
                    print(len(max_cycle))

            else:

                # For huge graphs you only need an approximation: look for long cycles using DFS heuristics instead of enumerating all cycles
                visited = set()
                max_cycle = []
                for node in G.nodes():
                    stack = [(node, [node])]
                    while stack:
                        current, path = stack.pop()
                        for neighbor in G.successors(current):
                            if neighbor == path[0]:  # cycle found
                                if len(path) > len(max_cycle):
                                    max_cycle = path
                            elif neighbor not in path:
                                stack.append((neighbor, path + [neighbor]))

            max_cycle_length = len(max_cycle)

            attractor_states = G.nodes()

            states = max_cycle
            for i, s in enumerate(attractor_states):
                if s not in max_cycle:
                    states.append(s)
        else:

            states  = G.nodes
            max_cycle_length = None

        return states, max_cycle_length, G

    
    """
    Generate a heatmap plot of a Boolean network attractor with a given attractor state.

    Args:
        attractor_state (str):
            Binary string representing the attractor state of the attractor for which
            the heatmap will be generated.

        extractMaxCycle (bool, optional):
            If True, the states in the resulting heatmap are partitioned into two sections 
            separated by a vertical solid orange line. The states on the left are ordered 
            according to the longest simple cycle in the attractor's state transition system. 
            The remaining states not in the cycle are listed on the right. Defaults to False.

        save_to (str, optional):
            Path to the file where the attractor heatmap image will be saved. Default
            value is None.

        plotAttractorSTS (bool, optional):
            If True, also plot the state transition system (STS) of the attractor.

        sts_filename (str, optional):
            Path to the file where the PyGraphviz image of the attractor's state
            transition system will be saved. Used only if `plotAttractorSTS` is True.

        title (str, optional):
            Title for the heatmap. If None, the default title will be added.

        verbose (bool, optional):
            If True, verbose mode is on.

    Returns:
        None
    """
    def plotAttractorAsHeatmap(self,
                               attractor_state: str,
                               extractMaxCycle: bool = False,
                               save_to: str = None,
                               plotAttractorSTS: bool = False,
                               sts_filepath: str = None,
                               title: str = None,
                               verbose: bool = False):

        states, max_cycle_length, G = self._getAttractorStatesForHeatmap(attractor_state, extractMaxCycle=extractMaxCycle, verbose=verbose)

        if plotAttractorSTS:

            if sts_filepath is None:
                raise ValueError("Must provide 'sts_filepath' when 'plotAttractorSTS' is set to True.")

            label_mapping = dict()
            for i, s in enumerate(states):
                label_mapping[s] = i
            nx.relabel_nodes(G, label_mapping, copy=True)

            G_agraph = nx.drawing.nx_agraph.to_agraph(G)
            G_agraph.layout('dot')

            # Creating folders if necessary
            if sts_filepath[-1] == '\\' or sts_filepath[-1] == '/':
                sts_filepath = sts_filepath[:-1]

            folder = os.path.dirname(sts_filepath)
            filename = os.path.basename(sts_filepath)

            if folder != '' and not os.path.exists(folder):
                os.makedirs(folder)

            G_agraph.draw(os.path.join(folder, filename)) # G_agraph is a instant of the PyGraphviz.AGraph class


        # Convert states to a DataFrame
        df = pd.DataFrame([tuple(bin2state(s)) for s in states], columns=self.node_names)
        if verbose:
            print(df)

        # Create a heatmap
        #if plotAttractorSTS:
            # fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        #    fig, axs = plt.subplots(2, 1, figsize=(int((11/42)*len(states)),int((6/7)*self.num_nodes)))
        #    ax = axs[0]
        #else:
        #fig, ax = plt.subplots(1, 1, figsize=(int((11/42)*len(states)),int((6/7)*self.num_nodes)))
        fig_heatmap = plt.figure(figsize=(max(int((11/42)*len(states)),1.5),max(int((6/7)*self.num_nodes),1.5)))
        ax = fig_heatmap.add_axes([0,0,1,1])
            
        sns.heatmap(df.T, annot=True, cbar=False, cmap="YlGn", linewidths=0.5, square=True, ax=ax)
        for i in range(len(states)):
            ax.axvline(x=i, linewidth=0.5, linestyle='--', color="black")
        for i in range(len(self.node_names)):
            ax.axhline(y=i, linewidth=0.5, color="grey")
        if extractMaxCycle:
            ax.axvline(x=max_cycle_length, linewidth=2, color="orange")
        ax.set_xlabel("Attractor State Index")
        ax.set_ylabel("Node names")
        ax.tick_params(axis='y', labelrotation=0)
        if title is None:
            ax.set_title("")
        else:
            ax.set_title(title)
        
        if save_to is not None:
            # Creating folders if necessary
            if save_to[-1] == '\\' or save_to[-1] == '/':
                save_to = save_to[:-1]

            folder = os.path.dirname(save_to)
            filename = os.path.basename(save_to)

            if folder != '' and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(os.path.join(folder, filename), bbox_inches='tight')

        plt.show()

        # if plotAttractorSTS:
        #     img = mpimg.imread(sts_filepath)
        #     # axs[1].set_xticks([])
        #     # axs[1].set_yticks([])
        #     # _ = axs[1].imshow(img)

        #     fig_sts = plt.imshow(img)
        #     plt.axis('off')
        #     plt.show()

    
    def plotAttractorsAsHeatmap(self, node_order: list[str] = None, orientation: str = 'vertical', title: str = None):

        if node_order is not None:
            # assert set(node_order) == set(self.node_names), "The set of names in node_order does not match the set of node names."
            reordering_indices = [self.node_names.index(el) for el in node_order]

        attractors = self._enumerate_attractors(self.find_all_attractors())

        dfs_dict = dict()
        for attractor_key in attractors:
            attractor_state = list(attractors[attractor_key])[0]
            states, _, _ = self._getAttractorStatesForHeatmap(attractor_state, verbose = False)
            if node_order is None:
                df = pd.DataFrame([tuple(bin2state(s)) for s in states], columns=self.node_names)
            else:
                reordered_states = [tuple(bin2state("".join(s[i] for i in reordering_indices))) for s in states]
                df = pd.DataFrame(reordered_states, columns=node_order)
            dfs_dict[attractor_key] = df

        if orientation == 'vertical':
            combined_df = pd.concat([df.T for df in dfs_dict.values()], axis=1, keys=dfs_dict.keys())
        else:
            combined_df = pd.concat([df for df in dfs_dict.values()], axis=0, keys=dfs_dict.keys())

        # Create the heatmap
        fig, ax = plt.subplots(1, 1, figsize=(int(combined_df.shape[0]*1.5), int(combined_df.shape[1]*1.5)+2))
        sns.heatmap(combined_df, annot=True, cbar=False, cmap="YlGn", linewidths=0.5, square=True, ax=ax)
    
        # Add visual separators and styling
        current_x = 0
    
        # Add thin dashed lines separating every individual state and thin grey
        # lines separating individual nodes.
        if orientation == 'vertical':
            for i in range(combined_df.shape[1]):
                ax.axvline(x=i, linewidth=0.5, color="grey")
            for i in range(combined_df.shape[0]):
                ax.axhline(y=i, linewidth=0.5, linestyle='--', color="black")
        else:
            for i in range(combined_df.shape[1]):
                ax.axvline(x=i, linewidth=0.5, linestyle='--', color="black")
            for i in range(combined_df.shape[0]):
                ax.axhline(y=i, linewidth=0.5, color="grey")                
        
        # Add thicker solid lines to visually separate the different DataFrames
        for name, df in dfs_dict.items():
            current_x += len(df)  # len(df) gives the number of columns in df.T
            # Avoid drawing a separator at the very right edge of the plot
            if current_x < combined_df.shape[1]:
                if orientation == 'vertical':
                    ax.axvline(x=current_x, linewidth=3, color="black") 
                else:
                    ax.axhline(y=current_x, linewidth=3, color="black") 
        # # Add the max_cycle_length line if provided
        # if max_cycle_length is not None:
        #    ax.axvline(x=max_cycle_length, linewidth=1.5, color="red", linestyle='-.')

        # Format Labels
        if orientation == 'vertical':
            ax.set_xlabel("Attractor State Index")
            ax.set_ylabel("Node Names")
        else:
            ax.set_ylabel("Attractor State Index")
            ax.set_xlabel("Node Names")

        
        # Format the x-tick labels to show the specific DataFrame name and the state index
        # combined_df.columns contains tuples like ('Condition A', 0) due to the concat keys
        if orientation == 'vertical':
            xticklabels = [f"{col[0]}\n{col[1]}" for col in combined_df.columns]
            ax.set_xticklabels(xticklabels, rotation=0) # rotation=0 keeps text readable
    
        plt.tight_layout()
        plt.show()    


    """
    Returns the set of neighboring states for a given state under the current update mode 
    of the Boolean network.

    Args:
        state (State): The state for which the neighboring states will be computed.
        excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
            when computing the neighboring states.

    Returns:
        set[State]: A set of neighboring states.
    """
    def getNeighborStates(self, state: State, excludedNodes: list[str] = []) -> set[State]:
        neighborStates = []

        excludedNodesIndices = [self.node_names.index(exclNode) for exclNode in excludedNodes]

        substitutions = dict()
        for i, node_value in enumerate(state):
            substitutions[
                self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

        if self.mode == 'asynchronous':
            # Asynchronous update scheme
            for node_index, fun in enumerate(self.functions):
                if node_index not in excludedNodesIndices:
                    new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0

                    new_state = list(state)
                    new_state[node_index] = new_node_value

                    neighborStates.append(new_state)

        elif self.mode == 'synchronous':
            # Synchronous update scheme
            new_state = list(state)

            for node_index, fun in enumerate(self.functions):
                if node_index not in excludedNodesIndices:
                    new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                    new_state[node_index] = new_node_value

            neighborStates.append(new_state)

        return set(tuple(neighborState) for neighborState in neighborStates)


    """
    Extracts attractors from their compressed BDD representations and converts them into an explicit form.

    Args:
        all_attractors (list[Function]): A list of BDDs representing individual attractors. Each BDD encodes 
            only attractor states, without transitions between them.

    Returns:
        dict[str, set[str]]: A dictionary where each key represents an attractor and each value 
            is a set of its states. The keys are strings of the form 'Ai', where *i* is a consecutive number 
            starting from 0, identifying each attractor.
    """
    def _enumerate_attractors(self, all_attractors: list[Function]) -> dict[str, set[str]]:

        attractors = dict()

        for i, at in enumerate(all_attractors):
            attractor = set()
            for models in self.__bdd.pick_iter(at, care_vars=self.node_names):
                state_str = ''
                for node_name in self.node_names:
                    state_str += '1' if models[node_name] else '0'
                attractor.add(state_str)

            attractors['A' + str(i)] = attractor

        return attractors


    """
    Extracts attractors from their compressed BDD representations and converts them into an explicit form.

    Args:
        all_attractors (list[Function]): A list of BDDs representing individual attractors. Each BDD encodes 
            only attractor states, without transitions between them.

    Returns:
        dict[str, set[State]]: A dictionary where each key represents an attractor and each value 
            is a set of its states. The keys are strings of the form 'Ai', where *i* is a consecutive number 
            starting from 0, identifying each attractor.
    """
    def _enumerate_attractor_states(self, all_attractors: list[Function]) -> dict[str, set[State]]:

        attractors = dict()

        for i, at in enumerate(all_attractors):
            attractor = set()
            for models in self.__bdd.pick_iter(at, care_vars=self.node_names):
                state_lst = []
                for node_name in self.node_names:
                    state_lst.append(1 if models[node_name] else 0)
                attractor.add(tuple(state_lst))

            attractors['A' + str(i)] = attractor

        return attractors


    """
    Draws the state transition graph of the Boolean network using PyGraphviz.

    Args:
        filepath (str):
            Path to the file where the state transition graph image will be saved. 
            The file extension defines the output format.  
            Available formats include: 'canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot', 'fig', 'gd', 'gd2',
            'gif', 'hpgl', 'imap', 'imap_np', 'ismap', 'jpe', 'jpeg', 'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic',
            'plain', 'plain-ext', 'png', 'ps', 'ps2', 'svg', 'svgz', 'vml', 'vmlz', 'vrml', 'vtx', 'wbmp',
            'xdot', and 'xlib'.  
            (Note: not all formats may be available on every system depending on how Graphviz was built.)

        layout (str, optional):
            Layout algorithm to use for visualizing the state transition graph.  
            Available layouts from PyGraphviz: 'neato', 'dot' (default), 'twopi', 'circo', 'fdp', and 'nop'.

        highlight_attractors (bool, optional):
            Indicates whether attractors should be highlighted with colors.  
            If 'True', non-attractor states are colored with the default `'chartreuse'` color 
            (from 'mcolors.CSS4_COLORS'), while attractor states are colored either randomly from the palette
            or using the provided 'color_names' list.

        use_bdds (bool, optional):
            If 'True' (default), attractors are computed symbolically using Binary Decision Diagrams (BDDs).  
            Otherwise, the 'networkx.attracting_components' function is used.  
            This option is ignored if 'highlight_attractors' is 'False'.

        color_names (list[str], optional):
            List of color names from 'mcolors.CSS4_COLORS' to be used for attractors.  
            Colors are applied sequentially. If there are fewer colors than attractors, some attractors 
            will share the same color.  
            At least one color must be provided.  
            Ignored if 'highlight_attractors' is 'False'.

        transient_state_color (str, optional):
            Name of the color in 'mcolors.CSS4_COLORS' used for non-attractor states.  
            Default is 'chartreuse'.  
            Ignored if 'highlight_attractors' is 'False'.

        selected_state_groups (list[list[State]], optional):
            List of groups of states to highlight in the graph.  
            Each group is a list of states.

        selected_group_colors (list[str], optional):
            Color names for each group of selected states.  
            The number of colors must match the number of groups in 'selected_state_groups'.

    Returns:
        None
    """
    def draw_state_transition_graph(self,
                                    filepath: str,
                                    layout: str = 'dot',
                                    highlight_attractors: bool = True,
                                    use_bdds: bool = True,
                                    color_names: list[str] = None,
                                    transient_state_color: str = 'chartreuse',
                                    selected_state_groups: list[list[State]] = [],
                                    selected_group_colors: list[str] = [],
                                   ) -> None:
        RANDOM_COLORS = color_names is None

        if not RANDOM_COLORS:
            assert len(color_names) > 0, "The number of provided color names must be at least 1!"

        print("Drawing the state transition graph ...")

        Gbn = self.generateStateTransitionGraph()
        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn = nx.drawing.nx_agraph.to_agraph(Gbn)

        # Modify node fillcolor and edge color.
        Dbn.node_attr.update(color='darkblue', style='filled', fillcolor=transient_state_color)
        Dbn.edge_attr.update(arrowsize=1)

        if highlight_attractors:

            if use_bdds:
                attractors = self._enumerate_attractors(self.find_all_attractors())
            else:
                attractors = self.getAttractors()

            colors = mcolors.CSS4_COLORS

            # # Modify node fillcolor and edge color.
            # Dbn.node_attr.update(color='darkblue', style='filled', fillcolor=transient_state_color)
            # Dbn.edge_attr.update(arrowsize=1)

            if RANDOM_COLORS:
                by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for
                                name, color in colors.items())
                color_names = [name for hsv, name in by_hsv]
                color_start_ind = color_names.index('palegreen')

            for i, BN_key in enumerate(attractors.keys()):
                # Select a color for the next attractor
                if RANDOM_COLORS:
                    color_ind = random.choice(range(color_start_ind, len(color_names)))
                else:
                    color_ind = i % len(color_names)

                color = colors[color_names[color_ind]]

                # Color the attractor states
                for state in attractors[BN_key]:
                    n = Dbn.get_node(state)
                    n.attr['fillcolor'] = color

        # Coloring of selected states
        if len(selected_state_groups) > 0:

            assert len(selected_state_groups) == len(selected_group_colors), "Wrong number of colors in selected_group_colors!"

            for color, selected_group in zip(selected_group_colors, selected_state_groups):
                for selected_state in [state2bin(selected_state) for selected_state in selected_group]:
                    n = Dbn.get_node(selected_state)
                    n.attr['fillcolor'] = color
        
        Dbn.layout(layout)

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # Dpbn is a instant of the PyGraphviz.AGraph class
        Dbn.draw(os.path.join(folder, filename))

        print(f"Done. The state transition graph is saved to {os.path.join(folder, filename)}")


    """
    Constructs an interaction graph for the Boolean network, where each edge represents
    a dependence of the target node on the source node.

    The dependency is determined syntactically — that is, by checking whether the Boolean 
    function of the target node includes the variable corresponding to the source node — 
    and not semantically. In other words, the method does not verify whether the source 
    node is an essential variable of the target node's Boolean function.

    Returns:
        networkx.DiGraph: A NetworkX directed graph object representing the interaction graph.
    """
    #@cache
    def getStructureGraph(self) -> nx.DiGraph:

        G = nx.DiGraph()

        for node in self.node_names:
            G.add_node(node)

        edges_aux = []
        for i, f in enumerate(self.functions):
            for s in f.symbols:
                # edges_aux.append((s, self.list_of_nodes[i]))
                edges_aux.append((str(s), self.node_names[i]))

        G.add_edges_from(edges_aux)

        return G


    """
    Plots the structure graph of the Boolean network.

    Args:
        layout (str, optional): The layout algorithm used to position the nodes. 
            Possible values are: 'spring' (default), 'circular', 'forceatlas', 
            'planar', 'random', 'shell', and 'spectral'.

    Returns:
        None
    """
    def plot_structure_graph(self, layout: str = 'spring') -> None:
        struct_graph = self.getStructureGraph()

        match layout:
            case 'spring':
                pos = nx.spring_layout(struct_graph)
            case 'circular':
                pos = nx.circular_layout(struct_graph)
            case 'forceatlas':
                pos = nx.forceatlas2_layout(struct_graph)
            case 'planar':
                pos = nx.planar_layout(struct_graph)
            case 'random':
                pos = nx.random_layout(struct_graph)
            case 'shell':
                pos = nx.shell_layout(struct_graph)
            case 'spectral':
                pos = nx.spectral_layout(struct_graph)
            case _:
                print("Unknown layout specified. Using the default 'spring' layout.")
                pos = nx.spring_layout(struct_graph)

        nx.draw_networkx(struct_graph, pos=pos)

        # ax = plt.axes()
        # nx.drawing.nx_pylab.draw(struct_graph,ax=ax)

        plt.show()


    """
    Plots the structure graph of the Boolean network using PyGraphviz and saves the image 
    to a file. The file format is determined by the file extension.

    Args:
        filepath (str): Path to the file where the structure graph image will be saved.  
            The file extension determines the output format.  
            Supported formats include: 'canon', 'cmap', 'cmapx', 'cmapx_np', 'dia', 'dot', 
            'fig', 'gd', 'gd2', 'gif', 'hpgl', 'imap', 'imap_np', 'ismap', 'jpe', 'jpeg', 
            'jpg', 'mif', 'mp', 'pcl', 'pdf', 'pic', 'plain', 'plain-ext', 'png', 'ps', 
            'ps2', 'svg', 'svgz', 'vml', 'vmlz', 'vrml', 'vtx', 'wbmp', 'xdot', and 'xlib'.  
            (Note: not all formats may be available on every system, depending on how Graphviz 
            was built.)

        layout (str, optional): Layout algorithm used for visualizing the structure graph.  
            Available layouts are: 'neato', 'dot' (default), 'twopi', 'circo', 'fdp', and 'nop'.

    Returns:
        None
    """    
    def plot_structure_graph_pgv(self, filepath: str, layout: str = 'dot') -> None:
        struct_graph = nx.drawing.nx_agraph.to_agraph(self.getStructureGraph())
        struct_graph.layout(layout)

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # struct_graph is an instant of the PyGraphviz.AGraph class
        struct_graph.draw(os.path.join(folder, filename))


    """
    Generates an interactive visualization of the Boolean network structure in HTML format.

    Args:
        filepath (str): Path to the file where the interactive graph will be saved.  
            The output is an HTML file, so the filename must have the `.html` extension.
        node_size (int, optional): Size of the graph nodes. Defaults to 20.
        height (int, optional): Height of the graph in pixels. Defaults to 500.
        width (int, optional): Width of the graph in pixels. Defaults to 500.

    Returns:
        None
    """
    def plot_structure_graph_interactive(self, filepath: str, node_size: int = 20, height: int = 500, width: int = 500) -> None:

        g = Network(height=height, width=width, directed=True, notebook=True, cdn_resources='in_line')
        g.toggle_hide_edges_on_drag(False)
        g.toggle_physics(True)  # Triggering the toggle_physics() method allows for more fluid graph interactions
        #g.barnes_hut()
        g.repulsion()
        # g.show_buttons(filter_=['nodes', 'edges', 'physics'])
        g.show_buttons(filter_=['physics'])

        # structure_graph = self.getStructureGraph()
        # g.from_nx(structure_graph, default_node_size=node_size, )

        for node_name in self.node_names:
            g.add_node(node_name, label=node_name, labelHighlightBold=False, shape='circle')

        for i, f in enumerate(self.functions):
            for s in f.symbols:
                g.add_edge(str(s), self.node_names[i], physics=False, title=str(f))

        # Creating folders if necessary
        if filepath[-1] == '\\' or filepath[-1] == '/':
            filepath = filepath[:-1]

        folder = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        # The filename extension must be .html
        name, extension = os.path.splitext(filename)
        if extension != '.html':
            if extension == '':
                print('Warning: Filename without extension. The required .html extension is added.')
            else:
                print('Warning: Inappropriate filename extension. Replacing it with .html')
            filename = name + '.html'

        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        path_str = os.path.join(folder, filename)
        g.show(path_str)

        print(f"Interactive structure graph saved in HTML format to {path_str}")


    """
        Creates the BDD representation of the transition matrix T for the given block.

        Description:
        Assume the block has n variables (x_1, ..., x_n). Consider two distinct Boolean states 
        of the Boolean network (BN): x = (x_1,...,x_n) and y = (y_1,...,y_n) of the BN. 
        Then T(x, y) = True if and only if the BN can transition from x to y in one update step. 
        Thus, T is a Boolean function of 2n variables. To illustrate the construction, consider
        a three-node BN with the following update functions:

        f1 = bdd.add_expr('(x1 & ~x2) | x3')
        f2 = bdd.add_expr('x2')
        f3 = bdd.add_expr('x1 & x2')

        In asynchronous mode, there are three possible updates — one for each node:
        updating x_1, updating x_2, or updating x_3.

        Suppose node x_1 is being updated. Then:
        
            x1_next = f_1,
            x2_next = x_2,
            x3_next = x_3.
        
        For node x_1, we construct a Boolean expression defining the transition relation:            

        R_1(x1, x2, x3, x1_next, x2_next, x3_next) <=> ( ~(xor(x1_next, f1)) & ~(xor(x2_next, x2)) & ~(xor(x3_next, x3)) ).
        
        The expression ~xor(a,b) == True if and only if a == b, so indeed the Boolean expression
        on the right is true if and only if the transition is due to the update of node x_1 alone.

        Similarly R_2 and R_3 are defined for the cases where only x_2 or x_3 is updated.
        The full transition relation T is then given by:    
    
                                T = R_1 | R_2 | R_3

        Args: 
            nodes (list[str]): The list of block variables.
            nodes_next (list[str]): The list of successor variables, derived from 'nodes' by 
                appending the "_next" suffix.
            functions (list[Function]): The list of update functions corresponding to the block nodes.

        Returns:
            The BDD representation of the local transition matrix for the given block.
        
    """
    def __create_bdd_from_boolean_network(self,
                                          nodes: list[str],
                                          nodes_next: list[str],
                                          functions: list[Function]) -> Function:
        eq = lambda a, b: ~self.__bdd.apply('xor', a, b)

        # T represents the transition matrix of BN.
        T_loc = self.__bdd.false

        #print("Constructing BDD encoded transition system of the BN ...", end='', flush=True)
        #for i in tqdm(range(len(nodes))):
        for i in range(len(nodes)):
            # Initial conditions for R_i.
            R_i = self.__bdd.true

            for j in range(len(nodes)):
                if i != j:
                    # The case: xj_next = xj for j != i.
                    R_i &= eq(self.__bdd.var(nodes_next[j]), self.__bdd.var(nodes[j]))
                else:
                    assert len(functions[j].support.intersection({nodes_next[j]})) == 0, \
                        "The construction of the BDD representation of the transition relation failed due \
                        to improper variable naming: the variables nodes_next and the variables of the Boolean \
                        function have non-empty intersection!"
                    # Here R_i represents the logic: xi_next = f_i.
                    R_i &= eq(self.__bdd.var(nodes_next[j]), functions[j])

            T_loc |= R_i
            
            if not CUDD_LOADED:
                self.__bdd.collect_garbage()

        #print('done.', flush=True)

        return T_loc


    """
    Finds all strongly connected components (SCCs) of the Boolean network and prepares
    the auxiliary data structures (technical containers) required by the 'find_all_attractors'
    method.

    Args:
        verbose (bool, optional): If True, prints diagnostic and progress information. 
            Defaults to False.

    Returns:
        dict[int, tuple[set, set]]: A dictionary mapping each block number to a tuple containing:
            - SCC_nodes (set): The set of nodes belonging to the SCC.
            - block_control_nodes (set): The set of control nodes associated with the block.
    """
    def __create_blocks(self, verbose : bool = False) -> dict[int, tuple[set, set]]:
        G = self.getStructureGraph()
        SCC_generator = nx.strongly_connected_components(G)
        SCC = list(SCC_generator)
        self.number_of_blocks = len(SCC)
        blocks = dict()

        # Dict holding "node" -> "scc_numb" where it belongs.
        self.node_block_number = dict()
        for block_nr, scc in enumerate(SCC):
            # Technical.
            block_dict = dict([
                (node, block_nr) for node in scc
            ])

            # Node -> number of block.
            self.node_block_number.update(block_dict)

            # Creating block.
            control_nodes_for_scc = set()

            for scc_node in scc:
                parent_set = set(G.predecessors(scc_node))
                control_nodes_for_scc |= parent_set

            control_nodes_for_scc -= scc
            blocks[block_nr] = (SCC[block_nr], control_nodes_for_scc)

        self.block_parents = defaultdict(set)
        self.block_children = defaultdict(set)
        for block_nr, (_, control_nodes) in blocks.items():
            for control_node in control_nodes:
                # Update block_parents.
                parent_block = self.node_block_number[control_node]
                self.block_parents[block_nr].add(parent_block)

                self.block_children[parent_block].add(block_nr)

        if verbose:
            print(f"SCC-based decomposition of the structure graph results in the following blocks : {blocks}")

        return blocks


    """
    Computes attractors by explicitly constructing the state transition graph.

    This method was originally introduced for testing the implementation of the symbolic, 
    BDD-based attractor computation algorithm. It should be used only for small networks 
    due to its explicit and potentially memory-intensive nature.

    Args:
        excludeNodes (list[str], optional): A list of node names whose updates will be excluded 
            during the construction of the state transition graph.

    Returns:
        dict[str, list[State]]: A dictionary where each key represents an attractor and 
            each value is a list of its states. The keys are strings of the form 'Ai', 
            where *i* is a consecutive number starting from 0, identifying each attractor.
    """
    def getAttractors(self, excludedNodes: list[str] = []) -> dict[str, list[State]]:
        GBn = self.generateStateTransitionGraph(excludedNodes)

        attractors = dict()
        for attractor_index, attractor in enumerate(nx.attracting_components(GBn)):
            attractors['A' + str(attractor_index)] = attractor

        return attractors


    def sample_initial_states(self, num_initial_states: int) -> list[list[int,]] | None:

        # Sample initial states in accordance with the specified environmental conditions
        initial_states = None
        if self.ec_bdd is not None:

            initial_states =  [[random.choice([True, False]) for _ in range(self.num_nodes)] for _ in range(num_initial_states)]

            # ToDo: This implementation is expensive in terms of memory usage - to be improved in the future.
            valid_ec_configurations = []
            for valid_ec_conf in self.__bdd_ec.pick_iter(self.ec_bdd, care_vars=self.ec_node_names):
                valid_ec_configurations.append(valid_ec_conf)

            if len(valid_ec_configurations) > num_initial_states:
                selected_ec_confs = random.sample(valid_ec_configurations, num_initial_states)
            else:
                selected_ec_confs = [valid_ec_configurations[ind] for ind in np.random.choice(list(range(len(valid_ec_configurations))), num_initial_states)]

            for i in range(num_initial_states):
                for node_name in selected_ec_confs[i].keys():
                    node_ind = self.node_names.index(node_name)
                    initial_states[i][node_ind] = selected_ec_confs[i][node_name]

        return initial_states


    # @staticmethod
    # def getAttractorsMonteCarlo(file):
    #     pbn = bang.load_from_file(file, "assa")
    #     pbn._n_parallel = min(max(77, pbn.n_nodes * 10), 2 ** pbn.n_nodes - 1)
    #     pbn.device = "gpu"

    #     attractors = pbn.monte_carlo_detect_attractors(trajectory_length=1100, attractor_length=1300)
    #     print(attractors)


    """
    Detects pseudo-attractors using Monte Carlo simulations.

    Args:
        n_parallel (int): Number of parallel simulations to run.
        burn_in_len (int): Length of the initial trajectory segment to discard as random and 
            non-representative for pseudo-attractor detection.
        history_len (int): Length of the trajectory segment used to classify states into 
            pseudo-attractors.
        threshold (float): The threshold hyper-parameter for pseudo-attractor states detection.

    Returns:
        set[State]: A set of pseudo-attractor states detected by the Monte Carlo approach.
    """
    def getAttractorsMonteCarlo(
            self, n_parallel:int = -1, burn_in_len:int = 1100, history_len:int = 1300, threshold: float = 0.15
    ) -> set[State]:
        if n_parallel == -1:
            n_parallel = min(max(77, self.num_nodes * 10), 2 ** self.num_nodes - 1)
        var_indices = {var: i for i, var in enumerate(self.node_names)}
        parent_variables = [sorted([var_indices[var.__str__()] for var in f.symbols]) for f in self.functions]        
        # truth_tables = [[y for _, y in truth_table(self.functions_str[i],
        #                                            sorted([x.__str__() for x in self.functions[i].symbols],
        #                                                   reverse=True)
        #                                           )
        #                 ] for i in range(self.num_nodes)
        #                ]
        truth_tables = [[y for _, y in truth_table(self.functions_str[i],
                                                           sorted([x.__str__() for x in self.functions[i].symbols],
                                                                  key = lambda l: var_indices[l],
                                                                  reverse=True)
                                                          )
                                ] for i in range(self.num_nodes)
                               ]

        pbn = bang.PBN(self.num_nodes,
                       [1 for _ in range(self.num_nodes)],
                       [len(f.symbols) for f in self.functions],
                       truth_tables,
                       parent_variables,
                       [[1.] for _ in range(self.num_nodes)],
                       0.,
                       [],
                       n_parallel=n_parallel)
        pbn.device = "gpu"

        # Sample initial states for the trajectories in accordance with the specified environmental conditions
        initial_states = None
        if self.ec_bdd is not None:

            initial_states =  [[random.choice([True, False]) for _ in range(self.num_nodes)] for _ in range(n_parallel)]

            # ToDo: This implementation is expensive in terms of memory usage - to be improved in the future.
            valid_ec_configurations = []
            for valid_ec_conf in self.__bdd_ec.pick_iter(self.ec_bdd, care_vars=self.ec_node_names):
                valid_ec_configurations.append(valid_ec_conf)

            if len(valid_ec_configurations) > n_parallel:
                selected_ec_confs = random.sample(valid_ec_configurations, n_parallel)
            else:
                selected_ec_confs = [valid_ec_configurations[ind] for ind in np.random.choice(list(range(len(valid_ec_configurations))),n_parallel)]

            for i in range(n_parallel):
                for node_name in selected_ec_confs[i].keys():
                    node_ind = self.node_names.index(node_name)
                    initial_states[i][node_ind] = selected_ec_confs[i][node_name]


        MC_pseudattractor_states = pbn.monte_carlo_detect_attractors(trajectory_length = burn_in_len,
                                                                     attractor_length = history_len,
                                                                     threshold = threshold,
                                                                     initial_states = initial_states)

        pseudoattractor_states = set()
        for pa_state_enc in MC_pseudattractor_states:
            # s = tuple([1 if val else 0 for val in list(ps_state)])
            s = self._decode_bang_state(pa_state_enc)
            pseudoattractor_states.add(s)


        # === Correctness check - to be removed in the future ===
        # Correctness check of whether found pseudo-attractor states comply with environmental conditions
        # input nodes settings. To be removed in the future.
        if CHECK_CORRECTNESS:
            if (self.ec_bdd is not None) and (len(valid_ec_configurations) > 0):
                fixed_input_nodes = self.getInputNodeNames()
                fixed_value_input_nodes = dict()

                for node_name in fixed_input_nodes:
                    fixed_value = True
                    ec_conf = valid_ec_configurations[0]

                    if node_name in ec_conf:
                        v = ec_conf[node_name]
                    else:
                        fixed_value = False
                        break

                    for ec_conf in valid_ec_configurations:
                        if node_name in ec_conf:
                            if v != ec_conf[node_name]:
                                fixed_value = False
                                break
                        else:
                            fixed_value = False
                            break

                    if fixed_value:
                        fixed_value_input_nodes[node_name] = 1 if v else 0
                            
                for pa_state in pseudoattractor_states:
                    for fixed_node_name in fixed_value_input_nodes.keys():
                        v = fixed_value_input_nodes[fixed_node_name]
                        input_node_idx = self.node_names.index(fixed_node_name)
                        assert v == pa_state[input_node_idx]
        # === End of Correctness Check ===============================


        return pseudoattractor_states
    

    """
    BN_Realisation class constructore - initializes a BN_Realisation object.

    Args:
        list_of_nodes (list[str]): A list of node names in the Boolean network.
        list_of_functions (list[str]): A list of Boolean functions in the network. 
            The order of the functions must correspond to the order of the nodes.
        mode (str): The update scheme for the Boolean network. Must be either 
            'asynchronous' or 'synchronous'.
        verbose (bool, optional): If True, prints diagnostic and progress information. 
            Defaults to False.

    Returns:
        BN_Realisation: An instance of the BN_Realisation class.
    """
    def __init__(self,
                 list_of_nodes: list[str],
                 list_of_functions: list[str],
                 mode: str = "asynchronous",
                 environmental_conditions: str | None = None,
                 target_configuration: str | None = None,
                 verbose: bool = False) -> BNReal:

        if mode not in ['asynchronous', 'synchronous']:
            raise ValueError(f"Wrong update scheme: {mode}")

        # Mode of the model.
        self.mode = mode

        # Container holding names of genes and theirs scc block number they belong.
        self.node_block_number = None

        # Number of SCC's in the Boolean Network structure.
        self.number_of_blocks = None

        # Number of nodes.
        self.num_nodes = len(list_of_nodes)

        # Names of nodes.
        self.node_names = list_of_nodes

        # String representing the update rules.
        self.functions_str = list_of_functions

        # Holds the index of modified update function.
        self.old_index = -1

        # Holds bool_algebra.Symbols of nodes.
        self.list_of_nodes = []

        # Technical dict which holds the gene symbol and its index in the array "list_of_nodes".
        gen_symbol_index = dict()

        for index, node_name in enumerate(list_of_nodes):
            node = self.__bool_algebra.Symbol(node_name)
            self.list_of_nodes.append(node)
            gen_symbol_index[node] = index

        # Make sure that constant functions are represented by proper strings
        for i, fun in enumerate(list_of_functions):
            if fun == '0':
                list_of_functions[i] = 'False'
            elif fun == '1':
                list_of_functions[i] = 'True'
            else:
                pass

        self.functions = []
        for fun in list_of_functions:
            self.functions.append(self.__bool_algebra.parse(fun, simplify=True))
        
        # self.original_function_algebra = dict(zip(self.list_of_nodes, self.functions))
        # self.node_update_function_algebra = dict(zip(self.list_of_nodes, self.functions))

        # Go through all boolean functions in BN and for each edge save to the container
        # the index of source and target nodes. 
        self.edges_order = []
        for i, f in enumerate(self.functions):
            for s in f.symbols: # Returns the Symbol of the char.
                self.edges_order.append((gen_symbol_index[s], i))

        # Declare Boolean variables in the BDD manager
        # (current and next state variables for the Boolean Network)
        for node_name in list_of_nodes:
            self.__bdd.declare(node_name)
            self.__bdd.declare(node_name + "_next")

        # Adding BDD expressions representing boolean update functions.
        self.bdd_expressions = []

        # Creating a dict containing relation: (node_name : str -> update_function : bdd_Function).
        # This dict will be used for searching purpose.
        self.node_name_to_bdd_update_function = dict()

        for index, fun in enumerate(list_of_functions):
            self.bdd_expressions.append(self.__bdd.add_expr(fun))
            self.node_name_to_bdd_update_function[self.node_names[index]] = self.bdd_expressions[-1]

        self.next_variables = [
            var + "_next" for var in self.node_names
        ]

        # For the hybrid Tarjan algorithm.
        self.visited_map = dict()

        # Dicts for renaming in searching for the images.
        self.x_to_x_next = dict(zip(self.node_names, self.next_variables))
        self.x_next_to_x = dict(zip(self.next_variables, self.node_names))

        # Block decomposition of the BN.
        self.number_of_blocks = None
        self.node_block_number = None
        self.block_parents = None
        self.block_children = None

        # Blocks is a dict container. It holds the relation:
        # block_number -> (SCC : set of name_node, control_nodes : set).
        self.blocks = self.__create_blocks(verbose = verbose)

        # For forward edgetics - remember original functions to restore BN in the future.
        self.original_functions_str = self.functions_str.copy()
        self.original_functions = self.functions.copy()
        self.bdd_expressions_original = self.bdd_expressions.copy()
        self.node_name_to_bdd_update_function_original = self.node_name_to_bdd_update_function.copy()

        # Setting environmental conditions
        if environmental_conditions is None:
            self.ec_bdd = None
            self.ec_fixed_nodes = None
        else:

            self.ec_node_names = [str(node) for node in self.__bool_algebra.parse(environmental_conditions).symbols]

            if len(set(self.ec_node_names).difference(set(self.getInputNodeNames()))) != 0:
                print("WARNING: The environmental conditions specification contains nodes that are not input nodes!")
            if len(set(self.getInputNodeNames()).difference(set(self.ec_node_names))) != 0:
                print("WARNING: The environmental conditions specification leaves some input nodes unspecified!")

            for node_name in self.ec_node_names:
                self.__bdd_ec.declare(str(node_name))

            self.ec_bdd = self.__bdd_ec.add_expr(environmental_conditions)

            self.ec_fixed_nodes = backbone(environmental_conditions)
            if verbose:
                print(f"Fixed node values in the specified environmental conditions: {self.ec_fixed_nodes}")

        # Setting target configuration expression
        if target_configuration is not None:
            self.target_configuration = self.__bool_algebra.parse(target_configuration)
            if verbose:
                print(f"Target configuration: {self.target_configuration}")
        else:
            self.target_configuration = None


    def restore_original_BN(self):
        self.functions_str = self.original_functions_str.copy()
        self.functions = self.original_functions.copy()
        self.bdd_expressions = self.bdd_expressions_original.copy()
        self.node_name_to_bdd_update_function = self.node_name_to_bdd_update_function_original.copy()

    
    def get_edges_order(self):
        edges = dict()
        for index, (source, target) in enumerate(self.edges_order):
            edges[index] = f'{self.node_names[source]} -> {self.node_names[target]}'
        return edges


    def getNumEdges(self):
        return len(self.edges_order)


    """
    Technical method: Used to determine whether a block is an elementary block.

    An elementary block is defined as one that has no parent blocks (i.e., its set
    of parent blocks is empty).

    Args:
        all_parents (set): The set of parent blocks for the block to be checked.

    Returns:
        bool: True if the block is elementary (has no parent blocks), otherwise False.
    """
    def __is_elementary_block(self, all_parents: set) -> bool:
        return True if all_parents == set() else False

 
    """
    Creates a BDD logical expression corresponding to a given Boolean network state.

    Given a dictionary mapping node names to Boolean values, this function constructs 
    a Binary Decision Diagram (BDD) expression that represents that state.
    this function will create a BDD logical expression wich corresponds to that state. For example: 
    (1) for models = {"x1": 1, "x2": 0, "x3": 0} the output will be a BDD of the expression: (x1 & ~x2 & ~x3)
    (2) for models = {"x1": 1, "x2": 1, "x3": 1} the output will be a BDD of the expression: (x1 & x2 & x3). 
    This function is needed by the 'push_forward_with_BDD' method to create initial
    set of BDDs corresponding to the provided states for subsequent propagation
    through the BDD representing the entire Boolean network dynamics.

    Args:
        models (dict[str, int]): A dictionary mapping node names to Boolean values (0 or 1) 
        representing the state to be converted into a BDD expression.

    Returns:
        Function: A BDD expression corresponding to the Boolean state given in 'models'.
    """
    def __bdd_representation_of_boolean_state(self, models: dict[str, bool]) -> Function:
        # Initialise the construction of the rule, which finally will represent the state in the dictionary.
        rule = self.__bdd.true

        # Iterate over the dictionary representing Boolean values of given nodes and add
        # the corresponding variable or its negation to the rule depending on the value
        # of the node.
        for variable, value in models.items():
            if not value:
                rule = self.__bdd.apply("and", rule, ~self.__bdd.var(variable))
            else:
                rule = self.__bdd.apply("and", rule, self.__bdd.var(variable))
        
        return rule


    """
    For a given list of initial binary states this function will create to each of them 
    a bdd logical expression wich corresponds to that state. For example 
    Initial_Set = [[1, 0, 0], [1,1,1]]. Then the output will be the list 
    of bdd's of the expressions: (x1 & ~x2 & ~x3), (x1 & x2 & x3), i.e. array  
    [bdd(x1 & ~x2 & ~x3), bdd(x1 & x2 & x3)]. 
    This function is needed in method "push_forward_with_BDD" to create initial
    set of bdd's corresponding to the states given by the user for further propagation 
    through the BDD describing whole Boolean Network. 
 
    Args:
        Initial_Set (list[list[int], ...]): list of binary lists of the same length representing some Boolean states
                                            of the Boolean Network.
        direction (string):                 technical string which is "forward" if the output is served for 
                                            searching image of some state and "backward" for the pre image of 
                                            some Boolean state.
    Returns:
        list: list of bdd's expressions corresponding to the states given in 
              Initial_Set.
    """
    def __bdd_representation_of_boolean_states(self,
                                               Initial_Set: list[list[int]],
                                               direction="forward") -> list:
        if direction == "forward":
            variables = self.node_names
        elif direction == "backward":
            variables = self.next_variables
        else:
            raise ValueError("The direction argument must be 'forward' or 'backward'")

        expressions = []

        for state in Initial_Set:
            # Initial condition for the rule which will correspond to
            # the fixed state in Initial_Set.
            rule = self.__bdd.true

            # Go through bits and if the bit is 1 then add to the rule operand "& var_name".
            # If bit is zero then add to the rule operand "&~var_name.
            # Note: dd library allows us to use different method for that, but it is much safer
            # to do it with bdd.apply method as shown below. This method allows one to use for
            # example 'xor' operators while other methods in dd library does not.
            for i, bit in enumerate(state):
                if bit == 0:
                    rule = self.__bdd.apply('and', rule, ~self.__bdd.var(variables[i]))
                else:
                    rule = self.__bdd.apply('and', rule, self.__bdd.var(variables[i]))
            expressions.append(rule)

        return expressions


    """
        This function will find some attractor state reachable from the node v. This is the 
        main method for the "find_some_attractor_state_from_fixed_state" method below. 
        This is in fact the standard Tarjan algorithm with the following modifications:
        (1) We find the first SCC which turns out to be BSCC (bottom strongly connected coponent).
        In other words - the first SCC found by Tarjan DFS algorithm is an attractor of BN.
        (2) We work on BDD's here. Thus the current state v is not a boolean vector but its BDD
        representation.     

        Important: The BDD representation does not improve the Tarjan algorithm here! The only
        improvment is that one need just to find first SCC and then the algorithm is straightforward
        with BDD (see find_all_attractors method for further details).

        Args: 
            v (Function): the BDD representation of current state.
            T_loc: the BDD representing of transition matrix of the block.
            index (list[int]): is the one-element list which holds the index in 
                               standard Tarjan algorithm. The list simulates C pointers.
            attractor_node (list[Function]): Once an attractor state is found this list will keep it.
                                             Until an attractor state is unknown - this list remains
                                             to be empty.
            all_values: list of nodes representing nodes which currently are in the processed block. 
    """
    def __hybrid_tarjan_recursive(self, v: Function, T_loc: Function, index: list, attractor_node: list,
                                  all_values: list):
        if attractor_node:
            return

        v_index = index[0]
        v_lowlink = index[0]
        v_onStack = True
        self.visited_map[v] = [v_index, v_lowlink, v_onStack]
        index[0] += 1

        # Find all pairs (x,x_next) satisfying pre logic.
        # It means that x has to be v and x_next has to be
        # connected with x in BN.
        pre = v & T_loc

        # Once all pairs are found use the exist operator for
        # taking x_next from the pair (x, x_next)
        post = self.__bdd.exist(all_values, pre)

        # x_next is the BDD represented in the x_next variables
        # and so one need to rename them for the future usage.
        rename_vars = dict(zip([val + "_next" for val in all_values], all_values))
        reach = self.__bdd.let(rename_vars, post)

        # Visit all neighbours of v.
        for state in self.__bdd.pick_iter(reach, care_vars=all_values):
            state_bdd = self.__bdd_representation_of_boolean_state(state)

            if state_bdd not in self.visited_map:  # state_bdd is not yet visited.
                self.__hybrid_tarjan_recursive(state_bdd, T_loc, index, attractor_node, all_values)

                #  Checking if the DFS found attractor node already.
                if attractor_node:
                    self.visited_map.clear()

                    return

                self.visited_map[v][self._V_LOWLINK_INDEX] = min(self.visited_map[v][self._V_LOWLINK_INDEX],
                                                                 self.visited_map[state_bdd][self._V_LOWLINK_INDEX])
            elif self.visited_map[state_bdd][self._V_ON_STACK_INDEX] is True:
                self.visited_map[v][self._V_LOWLINK_INDEX] = min(self.visited_map[v][self._V_LOWLINK_INDEX],
                                                                 self.visited_map[state_bdd][self._V_INDEX])

        if self.visited_map[v][self._V_LOWLINK_INDEX] == self.visited_map[v][self._V_INDEX]:
            # Append node which lies in bscc.
            attractor_node.append(v)

            return


    """
        For a give boolean state this function will find some attractor state reachable from 
        it. It uses the Hybrid Tarjan algorithm (see __hybrid_tyrjan_recursive method above).

        Args:
            bdd_of_init_state (list[int]): the boolean vector representing the state.
            T_loc (Function): bdd representing the transition matrix of the processing block. 
                                The word "loc" means here the local transition matrix, i.e. we create
                                the local transition matrix for a given block (or some union of blocks). 
            all_values (list[string]): list containing names of the block nodes.

        Returns:
            A bdd representation of the attractor node reachable from the given state.
    """
    def find_some_attractor_state_from_fixed_state(self, bdd_of_init_state, T_loc, all_values):
        index = [0]
        attractor_node = []
        self.__hybrid_tarjan_recursive(bdd_of_init_state, T_loc, index, attractor_node, all_values)

        return attractor_node[0]


    """
        For a given BDD representation of the BN state (or states) this function
        will compute all states reachable from this state (or states) in one step. The output 
        will be also of the BDD representation. 

        Args:
            states (Function): the BDD representation of state or multiple states.  
        Returns:
            Function: the BDD representation of all states reachable from this state in one step 
                      through the BN. 
    """
    def __one_step_image_monolithic(self, states: Function) -> Function:
        #if direction == "forward":
        #    nodes = self.node_names # x
        #    rename_vars = self.x_next_to_x # x_next -> x
        #elif direction == "backward":
        #    nodes = self.next_variables # x_next
        #    rename_vars = self.x_to_x_next # x -> x_next
        #else:
        #    raise ValueError(f"Wrong direction: {direction}")

        #T = self.bdd_transition_matrix
        T_global = self.__create_bdd_from_boolean_network(self.node_names, self.next_variables, self.bdd_expressions)
        transition_image = T_global & states
        image = self.__bdd.exist(self.node_names, transition_image)
        image_renamed = self.__bdd.let(self.x_next_to_x, image)

        return image_renamed


    """
        For a given BDD representation of the BN state (or states) this function
        will compute all states reachable from this state (or states) in one step with respect to the local
        transition matrix, i.e. transition matrix restricted to the block nodes. The output 
        will be also of the BDD representation. We can compute the forward image or backward image (preimage).

        Args:
            state (Function): the BDD representation of state or multiple states.
            T_loc (Function): the BDD representing the transition matrix.
            all_block_variables (list(string)): list of all names in the processing block.
            forward_vars, backward_vars (dict): technical dictionaries for renaming the variables in the result
                                                image bdd.
            direction (str): string representing direction of the image.
        Returns:
            Function: the BDD representation of all states reachable from this state in one step 
                      through the BN. 
        """
    def _one_step_image(self,
                        T_loc: Function,
                        all_block_variables: list,
                        state: Function,
                        forward_vars,
                        backward_vars,
                        direction: str) -> Function:
        if direction == "forward":
            nodes = all_block_variables  # x
            rename_vars = forward_vars  # x_next -> x
        elif direction == "backward":
            nodes = [val + "_next" for val in all_block_variables]  # x_next
            rename_vars = backward_vars  # x -> x_next
        else:
            raise ValueError(f"Wrong direction: {direction}")

        transition_image = state & T_loc
        image = self.__bdd.exist(nodes, transition_image)
        image_renamed = self.__bdd.let(rename_vars, image)

        return image_renamed

    """
        This function will find all attractors of the BN block reachable from the all_space BDD. The algorithm works as follows:
        Each time (until the space of all valid states is not empty), using the hybrid_tarjan DFS
        algorithm, we find the attractor state. After that the situation is straightforward:
        by finding the forward and backward image we find whole attractor and the set of states to delete. 
        The complexity of the problem highly depends on the structure of BDD of the attractor and images.

        Args:
            all_space (Function): the BDD representing the set of all states where attractors have to be found.
                                  all_space is None by the default - it works for elementary blocks. Once one has 
                                  to find attractors of a non-elementary block this parameter will be set to the 
                                  parent attractor.
            T_loc (Function): the BDD representing the transition matrix of block / unit of blocks.
            all_block_variables (list): the list of nodes in the block / unit of blocks.

        Returns: 
            list[Function]: list of BDD's representations of all attractors. 
    """
    def __find_all_attractors_in_scc_block(self, T_loc,
                                           all_block_variables: list,
                                           all_space=None):
        all_block_attractors = []
        if not all_space:
            all_space = self.__bdd.true

        while all_space != self.__bdd.false:
            model = next(self.__bdd.pick_iter(all_space, care_vars=all_block_variables), None)
            bdd_rule_for_model = self.__bdd_representation_of_boolean_state(model)
            attractor_node = self.find_some_attractor_state_from_fixed_state(bdd_rule_for_model, T_loc,
                                                                             all_block_variables)

            whole_attractor = attractor_node
            remember_last_image = attractor_node

            x_next_to_x = dict(zip([val + "_next" for val in all_block_variables], all_block_variables))
            x_to_x_next = dict(zip(all_block_variables, [val + "_next" for val in all_block_variables]))

            while True:
                # Find one-step image.
                image = self._one_step_image(T_loc,
                                             all_block_variables,
                                             remember_last_image,
                                             x_next_to_x,
                                             x_to_x_next,
                                             direction="forward")

                # Update all_space by deleting the new-found image.
                all_space = all_space & (~image)

                # Add image to the temper attractor holder.
                whole_attractor_constructor = whole_attractor | image

                # Check if the image updated. If not then we already
                # have the attractor.
                if whole_attractor_constructor == whole_attractor:
                    # print("Whole attractor was found.", flush=True)

                    break

                # Update whole-attractor container and last found image.
                whole_attractor = whole_attractor_constructor
                remember_last_image = image

            # Add new-found attractor to the list.
            all_block_attractors.append(whole_attractor)

            # Prepare variables for backward image.
            rename_variable = x_to_x_next  # Creating a renaming dict.
            attractor_node_rename = self.__bdd.let(rename_variable, attractor_node)  # Rename vars in bdd.
            remember_last_image = attractor_node_rename  # Create temper container for last found image.
            whole_backward_image = attractor_node_rename  # Create container for the whole backward image.

            # print("Computing backward image")
            while True:
                # Find backward image of the set "remember_last_image"
                back_image = self._one_step_image(T_loc,
                                                  all_block_variables,
                                                  remember_last_image,
                                                  x_next_to_x,
                                                  x_to_x_next,
                                                  direction="backward")

                # The result of above command is again in x_next for the future while-loop
                # calling and so to update all_space we need to rename them again.
                rename_variables_back = x_next_to_x  # Rename back_image back in terms of x.
                back_image_renamed = self.__bdd.let(rename_variables_back, back_image)
                all_space &= (~back_image_renamed)  # Update all_space by removing back_image.

                # Update the temper backward image container.
                whole_backward_image_constructor = whole_backward_image | back_image

                # Check if we have found everything.
                if whole_backward_image_constructor == whole_backward_image:
                    break

                # Update containers.
                whole_backward_image = whole_backward_image_constructor
                remember_last_image = back_image

        return all_block_attractors

    """
        For a given set of nodes it will return the list of bdd rules which prevents this variables to change. 
        This simulates the asynchronous mode. 
    """
    def __create_non_movers_conditions_for_variables_set(self, variables_set: set):
        eq = lambda a, b: ~self.__bdd.apply('xor', a, b)
        R = []

        for var in variables_set:
            var_next = var + "_next"
            R_i = eq(self.__bdd.var(var_next), self.__bdd.var(var))
            R.append(R_i)

        return R

    """
        Technical function which merges two elementary blocks by extending theirs transition matrix and merging 
        their attractors.

        all_attractors (list[Function]): list of parents attractors of lenght n.
        T_merged (Function): bdd representing the transition matrix of the parent block.
        T_block (Function):  bdd representing the transition matrix of the current elementary block.
        block_attractors (list[Function]): list of the attractors of the block of lenght m.
        block_variables (set): set holding names of the block variables.

        Returns:
            (T_new_merged, merged_attractors): (Function, list[Function] of the length n * m). 
    """
    def __merge_elementary_blocks(self, all_attractors: list[Function],
                                  T_merged: Function,
                                  all_variables: set,
                                  T_block: Function,
                                  block_attractors: list[Function],
                                  block_variables: set):
        # Firstly extend T_merged and T_block to whole set of variables block_variables + all_variables.
        R_all_variables = self.__create_non_movers_conditions_for_variables_set(all_variables)
        R_block_variables = self.__create_non_movers_conditions_for_variables_set(block_variables)

        T_block_extend = T_block
        for R_i in R_all_variables:
            T_block_extend &= R_i

        T_all_variables_extend = T_merged
        for R_i in R_block_variables:
            T_all_variables_extend &= R_i

        # Cross all attractors - because they are independent one can join them by using and operator.
        merged_attractors = []
        for at1 in all_attractors:
            for at2 in block_attractors:
                merged_attractors.append(at1 & at2)

        return T_block_extend | T_all_variables_extend, merged_attractors

    """
    Compute all attractors of a given Boolean Network (BN).

    This function implements one of the core and most technically complex features of the library — 
    the attractor detection algorithm based on the **block-split technique**.  
    Below we describe the method in detail.

    ---

    **Algorithm Overview**

    The algorithm identifies attractors by decomposing the Boolean Network into 
    **strongly connected components (SCCs)**.  
    Each SCC is treated as a separate **block**, and attractors are computed progressively, 
    starting from elementary (independent) blocks and extending their dynamics to dependent ones.

    1. **Decomposition into Blocks**
       - The Boolean Network is decomposed into SCCs (blocks).
       - Blocks that have no parent dependencies are called *elementary blocks*.
       - These blocks can be analyzed independently, so attractors are first computed there.

    2. **Extension to Dependent Blocks**
       - Once attractors for an elementary block are found, each dependent (child) block is processed.
       - For every attractor in the parent block, its dynamic behavior is extended to the child block,
         forming combined transition dynamics.

    ---

    **Detailed Example**

    Consider a Boolean Network consisting of two blocks:

         ___________              ____________
        | x1 <-> x2 |   B1       |  x3   x4  |   B2
        |   \  /    |            |    \   /  |
        |    x3 ----|------------|----> x5   |
        |___________|            |___________|

    **Step 1:**  
    Find attractors for block **B1**.  
    Since B1 has no parent blocks, it can be analyzed independently.  
    Let `T_B1` denote the transition matrix for B1, and let it have attractors `A1` and `A2`.

    **Step 2:**  
    Once B1 is processed, take its child block — **B2** (and enqueue any other children if they exist).

    **Step 3:**  
    Fix one attractor, for example `A1`, and derive its dynamics:  
    `dynamic_A1 = T_B1 & A1`  
    That is, restrict transitions in `T_B1` to those states belonging to attractor `A1`.

    **Step 4:**  
    Construct a new transition matrix for the merged block (B1 + B2).  
    Two kinds of node updates can occur:
    - Updates within B1 (`x1`, `x2`, `x3`) — follow the dynamics of `dynamic_A1`.
    - Updates within B2 (`x3`, `x4`, `x5`) — follow their respective Boolean functions.

    Note that node `x5` (from B2) depends on node `x3` (from B1).  
    Thus, when evaluating its Boolean function, `x3` can only take values 
    reachable within the attractor `A1`.  
    For example, if `A1 = {(1,1,1), (1,0,1)}`, then `x3 = 1` always, 
    so the input to `x5` must reflect that.

    Formally, we define:

        T_new = dynamic_A1 | (T_B2 & A1)

    where:
    - `dynamic_A1` corresponds to transitions within B1,
    - `T_B2 & A1` restricts transitions in B2 to valid inputs derived from `A1`.

    **Step 5:**  
    Using the combined transition matrix `T_new`, compute attractors for the merged block (B1 + B2).  
    The exploration space is limited to states consistent with `A1`.

    ---

    **Summary**
    This hierarchical approach — decomposing the BN into SCC blocks, 
    computing local attractors, and extending dynamics step-by-step — 
    significantly reduces computational complexity and enables efficient 
    computation of global attractors for large Boolean Networks. This algorithm 
    uses also BDD (Binary Decision Diagram) speeding up the computational process 
    and enables to work with large BN models.
    
    Args:
        all_space_constraints (Function): the BDD representing additional constraint on the set of states 
                                          where attractors are going to be found. If not None then 
                                          attractors found in BN will be cut to this space. Needed in 
                                          forward_edgetics method.
        path_to_file (string): string representing the path to the file where new found attractors are going 
                               to be saved. If this string is None then results are going to be saved in the 
                               current directory.
        filename (string): string representing the file name where new found attractors are going to be saved.
                           If None then results are going to be saved to two different files:
                           "attractors_dict_representation.txt" and "attractors_list_representation.txt".
        verbose (bool):     set to True to print details on the computation of attractors.
    Returns: 
        all_attractors (list[Function]): list of bdd's representing all attractors of the BN. 
                                         If all_space_constraint is not None then the result will be
                                         additionally cut to that space. 
    """
    def find_all_attractors(self, all_space_constraints: Function = None,
                            path_to_file: str = None,
                            filename: str = None,
                            verbose: bool = False) -> list[Function]:
        
        start_time = time.time()
        sys.setrecursionlimit(self._RECURSION_LIMIT)

        if verbose:
            print("Finding all attractors", flush=True)
            print(f"The network has {self.number_of_blocks}", flush=True)

        # Prepare containers for the further usage.
        q = deque(
            [
                block_nr
                for block_nr, (_, control_nodes) in self.blocks.items()
                if control_nodes == set()
            ]
        )
        processed_blocks_numbers = set()
        all_variables = set()
        all_attractors = []  # Will have the transit matrix T_loc, and attractor A.

        while q:
            block_nr = q.popleft()

            if block_nr in processed_blocks_numbers:
                continue

            # Find all children of the given block.
            all_control_nodes = self.blocks[block_nr][self._GET_CONTROL_NODES]
            all_parents = set([
                self.node_block_number[node_name] for node_name in all_control_nodes
            ])
            block_variables = self.blocks[block_nr][self._GET_SCC]

            # To correctly find update functions one has to
            # sort variables with respect to the general order.
            correct_order_of_block_variables = []
            for gen in self.node_names:
                if gen in block_variables:
                    correct_order_of_block_variables.append(gen)

            all_block_children = self.block_children[block_nr]
            block_nodes_next = [
                var + "_next" for var in correct_order_of_block_variables
            ]
            block_update_functions_bdd = [
                self.node_name_to_bdd_update_function[node_name] for node_name in correct_order_of_block_variables
            ]

            # All parent blocks are already processed.
            if all_parents.issubset(processed_blocks_numbers):

                # If the block is elementary block.
                if self.__is_elementary_block(all_parents):
                    if verbose:
                        print(f"The elementary block nr. {block_nr} is processed")

                    # Find transition matrix of the block.
                    T_loc = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                   nodes_next=block_nodes_next,
                                                                   functions=block_update_functions_bdd)

                    # With this transition matrix find all attractors of the elementary block.
                    block_attractors = self.__find_all_attractors_in_scc_block(T_loc,
                                                                               correct_order_of_block_variables,
                                                                               all_space=None)

                    # Mark the block as already processed.
                    processed_blocks_numbers.add(block_nr)

                    # If that is the first elementary block then just save the result:
                    # i.e. save transition matrix and the attractor states.
                    if not all_attractors:
                        all_attractors = [(T_loc, block_attractors)]
                    else:
                        # If the Boolean Network (BN) contains more than one elementary block,
                        # the results from these blocks must be merged.
                        #
                        # The merging procedure is straightforward:
                        # 1. Apply a logical "OR" operation to the transition matrices of the two elementary blocks.
                        #    Before doing so, extend each matrix to include all variables from both blocks —
                        #    this step preserves the asynchrony of the system.
                        # 2. For each attractor A from the previous elementary block,
                        #    combine it with each attractor B from the current elementary block
                        #    by computing their product (A & B).
                        if verbose:
                            print("Merging elementary blocks")
                        T_elementary_1, attractors_elementary_1 = all_attractors[0]
                        T_merged, attractors_merged = self.__merge_elementary_blocks(attractors_elementary_1,
                                                                                     T_elementary_1,
                                                                                     all_variables,
                                                                                     T_loc,
                                                                                     block_attractors,
                                                                                     block_variables)

                        # Save merged result to the general container.
                        all_attractors = [(T_merged, attractors_merged)]

                    # Mark block variables as already processed.
                    all_variables |= block_variables
                else:  # The block is not elementary.
                    if verbose:
                        print(f"The non-elementary block nr {block_nr} is processed")

                    # Create transition matrix for non-elementary block. At this stage it does not matter
                    # the dynamic of its control nodes.
                    T_child = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                     nodes_next=block_nodes_next,
                                                                     functions=block_update_functions_bdd)

                    # Extend transition matrix of the block to all variables. Similarly extend transition matrix
                    # of parent block to the block variables.
                    extend_conditions_for_new_block = self.__create_non_movers_conditions_for_variables_set(
                        all_variables)
                    extend_conditions_for_parent_block = self.__create_non_movers_conditions_for_variables_set(
                        block_variables)

                    # Each R_i represents the non-mover conditions for current block.
                    # More precisely by adding R_i we say that: variables from the parent block
                    # has to be fixed - it then preserves asynchronuous update mode.
                    for R_i in extend_conditions_for_new_block:
                        T_child &= R_i

                    # Find all new attractors with constructing realisations of the block with respect
                    # to the parent attractor and its dynamic.
                    new_all_attractors = []

                    # Iterate all attractors and theirs dynamics.
                    for (T_parent, attractor_list_parent) in all_attractors:
                        T_extend_parent = T_parent

                        # Similarly to the above code - we extend transition matrix for the parent
                        # on the child's variables.
                        for R_i in extend_conditions_for_parent_block:
                            T_extend_parent &= R_i

                        # For all attractors found under T_parent transition matrix create an realisation.
                        for parent_attractor in attractor_list_parent:
                            # Cut T_child to the previously found attractor.
                            T_child_extend = T_child & parent_attractor

                            # Find transition graph of the attractor.
                            attractors_dynamic = T_extend_parent & parent_attractor

                            # Create new merged transition matrix.
                            T_new = T_child_extend | attractors_dynamic

                            # Extend all_variables and processed_blocks_number containers.
                            all_variables |= block_variables
                            processed_blocks_numbers.add(block_nr)

                            # Find all attractors of the new merged blocks with the new
                            # transition matrix.
                            new_attractors = self.__find_all_attractors_in_scc_block(T_loc=T_new,
                                                                                     all_block_variables=list(
                                                                                         all_variables),
                                                                                     all_space=parent_attractor)
                            # if verbose:
                            #    print(f"Attractors of the block {block_nr} were found")

                            # Save the result.
                            new_all_attractors.append((T_new, new_attractors))

                    # Update all_attractors - this container keeps now all attractors together with theirs
                    # transition matrices of the new merged block.
                    all_attractors = new_all_attractors.copy()

                # Once the block is processed add all its block children to the queue q.
                for child in all_block_children:
                    q.append(child)

            # The block has parent which is not processed yet. Then put it on the end of
            # the queue.
            else:
                q.append(block_nr)

        # PRINTING STAFF.
        end_time = time.time()
        if verbose:
            print(f"All attractors are found in {int((end_time - start_time) / 60)} minutes")
        counter = 1

        if path_to_file or filename:        
            if not path_to_file:
                path_to_file = os.getcwd()
                print("The path to save results is not given. Results "
                    "are going to be saved in the current directory.\n",
                    flush=True)
            if not filename:
                print("The name of the file is not given. Results are going to be saved to the "
                    "attractors_dict_representation.txt for the dict representation and "
                    "to attractors_list_representation.txt "
                    "for a list representation.", flush=True)
                filename_1 = "attractors_dict_representation.txt"
                filename_2 = "attractors_list_representation.txt"
            else:
                filename_1 = filename + "_dict_representation.txt"
                filename_2 = filename + "_list_representation.txt"

            with open(os.path.join(path_to_file, filename_1), "w") as f, open(os.path.join(path_to_file, filename_2), "w") as g:
                g.write(f"The order is {self.node_names}")
                # all_attractors = [(T, A1), (T2, A2), .... , ], Ti - transition matrix of
                # merged blocks realisation, Ai - the final attractor in this realisation.
                for _, attractor_list in all_attractors:
                    for attractor in attractor_list:
                        f.write(f"Attractor {counter}:\n")
                        g.write(f"Attractor {counter}:\n")
                        f.write("=======================\n")
                        g.write("=======================\n")
                        print(f"Attractor nr {counter}", flush=True)
                        counter += 1
                        print(f"=======================")
                        states_counter = 0
                        # models: {name1 : True, name2 : False ,..., name_n : False}
                        for models in self.__bdd.pick_iter(attractor, care_vars=self.node_names):
                            state_string_dict = "{"
                            state_string_list = "("

                            for node_name in self.node_names:
                                name, value = node_name, int(models[node_name])
                                boolean_state = "1, " if models[node_name] else "0, "
                                state_string_dict += name + " : " + boolean_state
                                state_string_list += boolean_state

                            states_counter += 1
                            state_string_dict = state_string_dict[:-2]
                            state_string_list = state_string_list[:-2]
                            state_string_dict += "}"
                            state_string_list += ")"
                            f.write(state_string_dict + "\n")
                            g.write(state_string_list + "\n")
                            print(state_string_list, flush=True)

                        print(f"Number of states: {states_counter}\n"
                            f"-------------------", flush=True)
                        f.write(f"Number of states {states_counter} \n")
                        g.write(f"Number of states {states_counter} \n")

        # Returning attractors: one has to go through list of tuples and append
        # only bdd's representing attractors.
        remember_atractors = []
        for _, attractor_list in all_attractors:
            for attractor in attractor_list:
                if all_space_constraints:
                    intersection = attractor & all_space_constraints

                    if intersection != self.__bdd.false:
                        remember_atractors.append(intersection)
                else:
                    remember_atractors.append(attractor)

        return remember_atractors
    

    """
    Removes the dependency of "target_node" on "source_node" from the respective
    Boolean update function.

    This method removes the "source_node" from the Boolean update function
    associated with the "target_node". It is used to modify a Boolean network
    for forward-edgetics.

    Note:
        This method is deprecated. It implements an initial version of the 
        edge-removal algorithm (single edge removal for a given target node) 
        which checks individual "contexts" to determine the role of the source 
        node (activator or inhibitor). 
        
        Please use the `remove_edges()` method instead, which generalises 
        the edge-removal algorithm to multiple edges of a single target node 
        and provides a more efficient implementation.

    Args: 
        source_node (str): The name of the source node whose Boolean update
            function will be modified.
        target_node (str): The name of the node to be removed from the Boolean
            update function.
        modify_model (bool, optional): A flag indicating whether the edge is 
            removed from the Boolean network (`True`) or if only the new 
            function is printed (`False`). Defaults to True.
        verbose (bool, optional): If True, prints detailed additional detailed
            information to the console during execution. Defaults to False.

    Returns:
        str: A string representing the modified Boolean update function.
    """
    @deprecated("This method has been deprecated, use remove_edges instead.")
    def remove_edge(self, source_node: str, target_node: str, modify_model: bool = True, verbose: bool = False) -> str:

        # Function which returns index such that list_of_names[index] = name_to_find.
        def find_index_in_names(list_of_names, name_to_find) -> int:
            index = -1

            for i, node_name in enumerate(list_of_names):
                if node_name == name_to_find:
                    index = i
                    break

            if index == -1:
                raise ValueError(f"The source node {name_to_find} does not exist.")

            return index

        # Find indeces of source and removed nodes in the general node_names list.
        source_node_index_general = find_index_in_names(self.node_names, source_node)
        target_node_index_general = find_index_in_names(self.node_names, target_node)

        # Find function corresponding to the source_node.
        function_to_update = self.functions[target_node_index_general]

        # For technical future comparison - find symbol representing "removed_node".
        removed_symbol = self.list_of_nodes[source_node_index_general]

        # Take all symbols from the function. To avoid duplicates use set.
        function_variables = list(set(function_to_update.get_symbols()))

        # Find iremove_edgendex of "removed_node" in functions variables.
        removed_symbol_index_in_fun_var = find_index_in_names(function_variables, removed_symbol)

        # Save old function and its general index to the class variables.
        self.original_function = self.functions_str[target_node_index_general]
        self.old_index = target_node_index_general

        # This variable will keep new function after variable removing.
        updated_function = self.__bool_algebra.FALSE

        # Prepare loop range.
        n = len(function_variables) - 1

        if n == 0:
            value_1 = function_to_update.subs({function_variables[0] : self.__bool_algebra.FALSE}, simplify=True)
            value_2 = function_to_update.subs({function_variables[0] : self.__bool_algebra.TRUE}, simplify=True)

            if value_1 == self.__bool_algebra.FALSE and value_2 == self.__bool_algebra.TRUE:
                value_of_removed_function = self.__bool_algebra.FALSE

            # Removed value is inhibitor.
            elif value_1 == self.__bool_algebra.TRUE and value_2 == self.__bool_algebra.FALSE:
                value_of_removed_function = self.__bool_algebra.TRUE

            # Removed value is not an activator and inhibitor
            else:
                value_of_removed_function = value_1

            updated_function = value_of_removed_function
        else:
            # This loop will generate all possible inputs to the function_to_update.
            for number in range(2 ** n):
                binary_representation = (bin(number))[2 : ].zfill(n)  # Extend boolean array to the length n.
                bin_list = list(binary_representation)

                # Substitute 1 by algebra.FALSE and 0 by algebra.TRUE for further substitutions.
                prepare_values = [
                    self.__bool_algebra.FALSE if bit == '0' else self.__bool_algebra.TRUE
                    for bit in bin_list
                ]

                # To understand this if, else logic we are going to demonstrate the example:
                # Assume F(x1,x2,x3,x4) is our function to update. This loop will generate
                # only boolean table of length 3. For example prepare_value = [false, true, false].
                # We have two technical considerations x_remove variable index is 4
                # (i.e. Tilde F(x1,x2,x3) = remove(F, x4)) or x_remove variable index is < 4.
                # Assume without loss of generality the second case where x_remove index < 4,
                # let for example x_remove_index = 2. We then want to create two vectors:
                # [false, TRUE, false, true] and [false, FALSE, false ,true]. More precisely
                # we extend array by the value stored in x_remove_index and then we put True and False
                # in the place of x_remove_index. In that way we are always going to generate
                #
                #                     |                     [x_1, x_2, ..., TRUE, ..., x_n, x_remove]
                #                    \ /                  /
                #  [ x_1, x_2, ..., x_remove, ..., x_n] ->
                #                                         \
                #                                           [x_1, x_2, ..., FALSE, ..., x_n, x_remove]
                if removed_symbol_index_in_fun_var < n:
                    prepare_values.append(prepare_values[removed_symbol_index_in_fun_var])
                else:
                    prepare_values.append("TECHNICHAL APPEND")  # To extend the length of prepare_values.

                # Firstly set FALSE in the removed variable and prepare dict to the substitution.
                prepare_values[removed_symbol_index_in_fun_var] = self.__bool_algebra.FALSE
                substitutions = dict(zip(function_variables, prepare_values))
                value_1 = function_to_update.subs(substitutions, simplify=True)

                # Now set TRUE into removed variable.
                substitutions[removed_symbol] = self.__bool_algebra.TRUE
                value_2 = function_to_update.subs(substitutions, simplify=True)

                # Apply forward_edgetics algortithm to find value of function
                # basing on the above value_1, value_2.
                # value_of_removed_function = self.__bool_algebra.FALSE

                # Removed variable is activator.
                if value_1 == self.__bool_algebra.FALSE and value_2 == self.__bool_algebra.TRUE:
                    value_of_removed_function = self.__bool_algebra.FALSE

                # Removed variable is inhibitor.
                elif value_1 == self.__bool_algebra.TRUE and value_2 == self.__bool_algebra.FALSE:
                    value_of_removed_function = self.__bool_algebra.TRUE

                # Removed variable is neither an activator nor an inhibitor.
                else:
                    value_of_removed_function = value_1

                # This fragment will create a boolean expression corresponding
                # to the variable values. For example if for
                # x_1 = true, x_2 = false, x_remove = false, x_4 = true we have
                # Tilde{F}(x_1, x_2, x_3) = True we do the following expression:
                # (x_1 & ~x_2 & x4)
                # which is true on exactly this vector.
                if value_of_removed_function == self.__bool_algebra.TRUE:
                    one_logical_block = self.__bool_algebra.TRUE

                    for i, val in enumerate(prepare_values):
                        if i != removed_symbol_index_in_fun_var:
                            if val == self.__bool_algebra.TRUE:
                                one_logical_block &= function_variables[i]
                            else:
                                one_logical_block &= ~function_variables[i]

                    # This logical blok is combining all expressions to the
                    # one function by putting "or" operator between "one_logical_block", i.e.
                    # the result will be: Tilde{F} = (expr_1) | (expr_2) | (expr_3) ... | (expr_n).
                    updated_function |= one_logical_block

        updated_function = updated_function.simplify()
        self.functions[target_node_index_general] = updated_function

        if verbose: 
            print("=============================================================================================")
            print(f"After removing the variable \"{source_node}\" from the function: \n"
                f"{function_to_update} \n"
                f"we get \n"
                f"{updated_function}")
            print("=============================================================================================")

        converted_to_string = str(updated_function)

        if modify_model:
            if converted_to_string == self._CONVERTED_FALSE_IN_BOOLEAN_PY_TO_STRING:
                self.functions_str[target_node_index_general] = str(self.node_names[target_node_index_general]) \
                                                                + "&~" \
                                                                + str(self.node_names[target_node_index_general])
                self.bdd_expressions[target_node_index_general] = self.__bdd.add_expr(self.functions_str[target_node_index_general])
            elif converted_to_string == self._CONVERTED_TRUE_IN_BOOLEAN_PY_TO_STRING:
                self.functions_str[target_node_index_general] = str(self.node_names[target_node_index_general]) \
                                                                + "|~" \
                                                                + str(self.node_names[target_node_index_general])
                self.bdd_expressions[target_node_index_general] = self.__bdd.add_expr(self.functions_str[target_node_index_general])

            else:
                self.functions_str[target_node_index_general] = str(updated_function)
                self.bdd_expressions[target_node_index_general] = self.__bdd.add_expr(str(updated_function))

            # Update of the remaining data structures
            self.original_function = None
            self.old_index = -1
            self.node_name_to_bdd_update_function[self.node_names[target_node_index_general]] = self.bdd_expressions[target_node_index_general]
            self.blocks = self.__create_blocks()

        return converted_to_string # Be aware of the string convertation of the True and False constants in boolean.py!


    """
    Short commentary: for the list of indexes remove all edges positioned on that indexes. 

    Return: None. Updates all "original" containers.
    """
    def remove_edges(self, edges_to_remove : list[int], verbose=True):
        # Step (1): go through edges_to_remove and create grouped_dict: target_index -> [begin_indexes].
        # Step (2): go through the dict created in the Step (1) and remove from the update function all
        # variables in the value list.

        grouped_dict = defaultdict(set)
        for edge_index in edges_to_remove:
            # Take indexes of the source and target of that edge.
            source_index, target_index = self.edges_order[edge_index]
            grouped_dict[target_index].add(source_index)

        new_functions_and_index = []
        for target_index, set_of_source_indexes in grouped_dict.items():
            if verbose:
                print(f"From the node {str(self.list_of_nodes[target_index])} remove nodes:",
                      f"{tuple([str(self.list_of_nodes[node]) for node in set_of_source_indexes])}")
                
            # Get updated function from the list of all functions.
            target_function = self.functions[target_index] 

            if verbose:
                print(f"-> Updated function: {str(target_function)}")

            # From the indexes create symbols of Boolean algebra
            # corresponding to the gene names.
            nodes_to_remove = [self.list_of_nodes[index] for index in set_of_source_indexes]

            # Create the array of FALSE values in the Boolean library.
            values = [self.__bool_algebra.FALSE] * len(nodes_to_remove)

            # Create dict of variable -> value for further insertion.
            substitutions = dict(zip(nodes_to_remove, values))

            # Evaluate and simplify boolean function by substitution new values.
            modified_fun = target_function.subs(substitutions, simplify=True)

            if verbose:
                print(f"-> After modification: {str(modified_fun)}")
            
            # Save the modified function and its index in the source array to the 
            # savior container.
            new_functions_and_index.append((target_index, modified_fun))

        # Create the list of modified Boolean functions.

        # Usprawnic to bo za duzy complexity bez sensu. Przeniesc do konstruktora.
        for fun_index, modified_fun in new_functions_and_index:
            self.functions[fun_index] = modified_fun

            if modified_fun == self.__bool_algebra.TRUE:
                modified_fun_str = 'True'
            elif modified_fun == self.__bool_algebra.FALSE:
                modified_fun_str = 'False'
            else:
                modified_fun_str = str(modified_fun)

            if modified_fun == self.__bool_algebra.TRUE:
                self.functions_str[fun_index] = '1'
            elif modified_fun == self.__bool_algebra.FALSE:
                self.functions_str[fun_index] = '0'
            else:
                modified_fun_str = str(modified_fun)

            self.bdd_expressions[fun_index] = self.__bdd.add_expr(modified_fun_str)
            self.node_name_to_bdd_update_function[self.node_names[fun_index]] = self.bdd_expressions[fun_index]


    """
        This function modifies a selected update function by removing its dependency on the removed_node.
        It then computes the n-step image of the initial state under the modified Boolean network and searches
        for all attractors reachable from that image.

        Args:
            source_node (string):               string representing the name of node (gene) from which the target_node
                                                dependency will be removed.
            target_node (string):               string representing the name of the removed node from the source node.
            number_of_steps (int):              integer representing the number of push_forward of the init_state via
                                                modified BN.
            init_states (State):                the tuple representing the initial state of BN from which the 
                                                analysis begins.
            verbose (bool):                     a flag indicating whether the function should run in verbose mode.

        Returns:
            list[Function]: the list of all attractors reachable from the push_forward(init_state, number_of_steps, modified BN).
    """
    def forward_edgetics(self,
                         edges_to_remove: list[int],
                         number_of_steps: int,
                         init_state: State,
                         verbose: bool = False) -> list[Function]:
        
        # Remove edges.
        self.remove_edges(edges_to_remove=edges_to_remove)

        # Create the copy of number of steps for further printing.
        n = number_of_steps

        models = dict(zip(self.node_names, init_state))

        # Create a BDD representation of init_state.
        bdds_of_init_states = self.__bdd_representation_of_boolean_state(models)

        # for state_bdd in bdds_of_init_states:
        #     bdd_of_init_states = bdd_of_init_states | state_bdd
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()

        # This BDD will hold whole push_forward image of initial state n-times.
        image = bdds_of_init_states
        
        # Technical container for checking if the image is not changing anymore.
        # Could be used if the BN is relatively small and n is big enough.
        last_image = bdds_of_init_states
        node_next = [name + "_next" for name in self.node_names]
        forward_vars = dict(zip(node_next, self.node_names))
        backward_vars = dict(zip(self.node_names, node_next))

        T_global = self.__create_bdd_from_boolean_network(self.node_names,
                                                          nodes_next=node_next,
                                                          functions=self.bdd_expressions)

        while n > 0:
            image = self._one_step_image(T_global,
                                         self.node_names,
                                         last_image,
                                         forward_vars=forward_vars,
                                         backward_vars=backward_vars,
                                         direction="forward")

            if last_image == image:
                break

            n -= 1
            last_image = image

        # Replace a modified update function to the original one after the remove_edges method.
        self.restore_original_BN()

        T_global = self.__create_bdd_from_boolean_network(self.node_names,
                                                          nodes_next=node_next,
                                                          functions=self.bdd_expressions)

        # Searching for the forward image.
        whole_forward_image = image
        remember_last_image = image

        while True:
            one_step_forward_image = self._one_step_image(T_global,
                                                          self.node_names,
                                                          remember_last_image,
                                                          forward_vars=forward_vars,
                                                          backward_vars=backward_vars,
                                                          direction="forward")
            forward_image_constructor = whole_forward_image | one_step_forward_image

            if whole_forward_image == forward_image_constructor:
                break

            whole_forward_image = forward_image_constructor
            remember_last_image = one_step_forward_image

        # Create all_space.
        all_space = whole_forward_image

        # These lines could be removed in the future.
        # if verbose:
        #     print("All image is then:", flush=True)
        #     # print(f"BEFORE REORDER {len(self.__bdd)}")
        #     self._print_bdd(all_space)
        #     # print(f"AFTER REORDER {len(self.__bdd)}")
        # ------------

        # bdd.reorder(self.__bdd, self.custom_order)
        # Find all atrractors in all_space.

        attractors = self.find_all_attractors(all_space_constraints=all_space)

        return attractors
    

    # For testing only - undocumented: Get the set of attractors reachable from the given set of states.
    def _get_reachable_attractors(self, states: list[tuple[int,...]], withTransientStates : bool = False):
        attractors = self.getAttractors()
        #attractors = self._enumerate_attractors(self.find_all_attractors())

        attractor_states = set()
        state2attr_dict = dict()

        for attractor_key in attractors.keys():
            attractor = attractors[attractor_key]
            attractor_states.update(attractor)

            for attrState in attractor:
                state2attr_dict[attrState] = attractor_key

        reachableAttractors = set()
        current_states = set(states)
        processed_states = set()
        while len(current_states) != 0:
            next_states = set()

            for state in current_states:
                next_states.update(self.getNeighborStates(state))
                processed_states.add(state)

            next_states_bin = set([state2bin(s) for s in next_states])
            for state_bin in next_states_bin.intersection(attractor_states):
                reachableAttractors.add(state2attr_dict[state_bin])
                next_states_bin.remove(state_bin)

            current_states = set([bin2state(s) for s in next_states_bin]).difference(processed_states)

        attractor_states = set()
        for attractor_key in reachableAttractors:
            attractor_states = attractor_states.union(attractors[attractor_key])

        if withTransientStates:
            transient_states_bin = set([state2bin(s) for s in processed_states]).difference(attractor_states)
            #return reachableAttractors, transient_states_bin
            return attractor_states, transient_states_bin

        #return reachableAttractors
        return attractor_states


    """

    """
    def get_reachable_attractors_with_bdd(self, states: list[tuple[int,...]]) -> dict[str, set[str]]:

        states_bdd = self.__bdd.false
        for state_bdd in self.__bdd_representation_of_boolean_states([list(s) for s in states]):
            states_bdd = states_bdd | state_bdd
        # print(f"Size of BDD manager before garbage collection: {len(self.__bdd)}")
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()
        # print(f"Size of BDD manager after garbage collection: {len(self.__bdd)}")

        # Searching for the forward image.
        whole_forward_image = states_bdd
        remember_last_image = states_bdd

        while True:
            # ToDo: Consider decomposition-based implementation (?)
            one_step_forward_image = self.__one_step_image_monolithic(remember_last_image)
            forward_image_constructor = whole_forward_image | one_step_forward_image

            if whole_forward_image == forward_image_constructor:
                break

            whole_forward_image = forward_image_constructor
            remember_last_image = one_step_forward_image

        #bdd.reorder(self.__bdd, self.custom_order)
        # Find all attractors in all_space.
        attractors = self.find_all_attractors(all_space_constraints=whole_forward_image)

        return self._enumerate_attractors(attractors)
    


    def get_reachable_states_in_n_steps(self, states: list[tuple[int,...]], num_steps: int = 1) -> dict[str, set[str]]:

        states_bdd = self.__bdd.false
        for state_bdd in self.__bdd_representation_of_boolean_states([list(s) for s in states]):
            states_bdd = states_bdd | state_bdd
        # print(f"Size of BDD manager before garbage collection: {len(self.__bdd)}")
        if not CUDD_LOADED:
            self.__bdd.collect_garbage()
        # print(f"Size of BDD manager after garbage collection: {len(self.__bdd)}")

        # Searching for the forward image.
        one_step_forward_image = states_bdd
        remember_last_image = states_bdd

        for _ in range(num_steps):
            # ToDo: Consider decomposition-based implementation (?)
            one_step_forward_image = self.__one_step_image_monolithic(remember_last_image)
            
            if remember_last_image == one_step_forward_image:
                break

            remember_last_image = one_step_forward_image

        states = set()
        for models in list(self.__bdd.pick_iter(remember_last_image, care_vars=self.node_names)):
            state = []
            for node_name in self.node_names:
                state.append(1 if models[node_name] else 0)
            states.add(tuple(state))

        return states


    def simulate_asynchronous(self, init_state: State, num_steps: int, verbose: bool=False) -> State:
        state = list(init_state)

        if CHECK_CORRECTNESS:
            assert len(state) == self.num_nodes, f"State length {len(state)} is different from the number of nodes ({self.num_nodes})!"
            assert len(state) == len(self.list_of_nodes)

        if verbose:
            print(f"Simulate path starting from the state: \n"
                  f"{state} \n"
                  f"Number of steps: {num_steps} \n"
                  f"The next steps are:")
        
        for _ in range(num_steps):
            substitutions = dict()

            for i, node_value in enumerate(state):
                substitutions[self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1\
                                                                 else self.__bool_algebra.FALSE

            # Choose randomly node to update.
            node_index = random.randrange(self.num_nodes)
            fun = self.functions[node_index]

            new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
            state[node_index] = new_node_value

            if verbose:
                print(f"-> {state}")

        return tuple(state)
    

    """
    """
    def get_reachable_attractors_with_edge_removed(self, source_node : str, target_node : str, n_steps : int, initial_state) -> dict[str, set[str]]:
        #reachable_attractors = []

        # Find the index of the specified edge in the self.edges_order list to provide it to the forward_edgetics method.
        edge_index = None
        
        try:
            source_index = self.node_names.index(source_node)
        except ValueError:
            raise ValueError("Wrong source_node name!")
        try:
            target_index = self.node_names.index(target_node)
        except ValueError:
            raise ValueError("Wrong target_node name!")
        
        for index, edge in enumerate(self.edges_order):
            if edge == (source_index, target_index):
                edge_index = index

        if edge_index == None:
            raise ValueError(f"Non-existing edge {(source_node, target_node)}!")

        # attr_from_s = self._enumerate_attractors(self.forward_edgetics(source_node, target_node, n_steps, initial_state))
        attr_from_s = self._enumerate_attractors(self.forward_edgetics([edge_index], n_steps, initial_state))

        #for attr in attr_from_s.values():
        #    reachable_attractors.append(attr)

        return attr_from_s
    

    def is_fixpoint_attractor(self, state: State) -> bool:
        substitutions = dict()
        for i, node_value in enumerate(state):
            substitutions[
                self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

        # Applying all functions one-by-one.
        for node_index, fun in enumerate(self.functions):
            new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
            if new_node_value != state[node_index]:
                return False

        return True


    def _is_fixpoint_test(self, state: State) -> list[int]:
        substitutions = dict()
        for i, node_value in enumerate(state):
            substitutions[
                self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1 else self.__bool_algebra.FALSE

        # Applying all functions one-by-one.
        l = []
        for node_index, fun in enumerate(self.functions):
            new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
            if new_node_value != state[node_index]:
                l.append(node_index)

        return l
    

    def _decode_bang_state(self, encoded_state: np.ndarray[np.uint32, ]):
    
        decoded_state = []

        num_parts = len(encoded_state) # math.ceil(self.num_nodes / 32)
        last_part_size = self.num_nodes % 32 if self.num_nodes % 32 > 0 else 32

        for part_num in range(num_parts):
            num_bits = last_part_size if part_num == num_parts - 1 else 32
            decoded_part = int2bin(encoded_state[part_num], num_bits)[::-1]
            for bit in decoded_part:
                decoded_state.append(1 if bit == '1' else 0)

        return bin2state(decoded_state)
    

    def simulate(self, nsteps: int, init_state: State | None = None, full_trajectory: bool = True) -> list[State]:
        var_indices = {var: i for i, var in enumerate(self.node_names)}
        parent_variables = [sorted([var_indices[var.__str__()] for var in f.symbols]) for f in self.functions]
        truth_tables = [[y for _, y in truth_table(self.functions_str[i],
                                                   sorted([x.__str__() for x in self.functions[i].symbols],
                                                          key = lambda l: var_indices[l],
                                                          reverse=True)
                                                  )
                        ] for i in range(self.num_nodes)
                       ]

        pbn = bang.PBN(self.num_nodes,
                       [1 for _ in range(self.num_nodes)],
                       [len(f.symbols) for f in self.functions],
                       truth_tables,
                       parent_variables,
                       [[1.] for _ in range(self.num_nodes)],
                       0.,
                       [],
                       n_parallel=min(max(77, self.num_nodes * 10), 2 ** self.num_nodes - 1))
        pbn._n_parallel = min(max(77, pbn.n_nodes * 10), 2 ** pbn.n_nodes - 1)
        pbn.device = "gpu"

        if init_state is not None:

            pbn.set_states([[True if bit==1 else False for bit in init_state]])

        else:

            init_state = [random.choices([True, False], k=self.num_nodes)]
            pbn.set_states(init_state)

        pbn.save_history = full_trajectory

        # Simulate the network for nsteps
        pbn.simple_steps(nsteps)

        # Access the simulation history
        # print("Trajectory history:", pbn.history)
        # Access the final state
        # print("Last state:", pbn.last_state)

        if full_trajectory:
            # trajectory = [int2bin(s[0][0], self.num_nodes)[::-1] for s in pbn.history]
            # trajectory = [bin2state(int2bin(s[0][0], self.num_nodes)[::-1]) for s in pbn.history]
            trajectory = [self._decode_bang_state(s[0]) for s in pbn.history]
        else:
            # assert pbn.history[-1][0][0] == pbn.last_state[0][0]
            # trajectory = [bin2state(int2bin(pbn.last_state[0][0], self.num_nodes)[::-1])]
            trajectory = [self._decode_bang_state(pbn.last_state[0])]

        return trajectory


    def _one_step_image_monolithic_with_direction(self, states: Function, direction: str = 'forward') -> Function:
        
        T_global = self.__create_bdd_from_boolean_network(self.node_names, self.next_variables, self.bdd_expressions)
        
        if direction == "forward":
            transition_image = T_global & states
            image = self.__bdd.exist(self.node_names, transition_image)
            image_renamed = self.__bdd.let(self.x_next_to_x, image)
            return image_renamed
        
        elif direction == 'backward':
            transition_image = T_global & self.__bdd.let(self.x_to_x_next, states)
            preimage = self.__bdd.exist(self.next_variables, transition_image)
            # preimage = self.__bdd.quantify(transition_image, self.next_variables, forall=False)
            return preimage
        
        else:
            raise ValueError(f"Wrong direction: {direction}")
    

    """
    Construct a BDD representing a given Boolean network state.

    The resulting BDD uses a set of variables corresponding to node names.

    Args:
        state (list[int]): Binary values (0 or 1) representing the network state,
            ordered according to the node list.

    Returns:
        BDD object (Function): A binary decision diagram representing the given state.
    """
    def get_BDD_representation(self, state: list[int]) -> Function:
        return self.__bdd_representation_of_boolean_states([state], direction='forward')[0]


    # def check_bdd_representation(self, f):
    #     state_set = set()
    #     for models in self.__bdd.pick_iter(f, care_vars=self.node_names):
    #         state_str = ''
    #         for node_name in self.node_names:
    #             state_str += '1' if models[node_name] else '0'
    #         state_set.add(state_str)

    #     return state_set

    """
    Compute a BDD representing the weak basin of an attractor given by one of its states.

    If 'state' belongs to an attractor, this returns the weak basin of that attractor.
    For an arbitrary state, this returns the set of all states that can reach 'state'
    through backward transitions (backward-reachable states).

    Args:
        state (list[int]): Binary values (0 or 1) representing the network state,
            ordered according to the node list.

    Returns:
        BDD: A binary decision diagram representing either the weak basin of the
            attractor or, more generally, the backward-reachable set of states.
    """
    def compute_weak_basin_BDD(self, state: list[int]) -> Function:

        current_preimage = self.__bdd_representation_of_boolean_states([state], direction='forward')[0]

        # remember_last_image = current_preimage # Create temper container for last found image.
        whole_backward_image = current_preimage # Create container for the whole backward image.

        while True:
            # Find backward image of the set "remember_last_image"
            current_preimage = self._one_step_image_monolithic_with_direction(current_preimage, direction="backward")

            # Update the temper backward image container.
            whole_backward_image_constructor = whole_backward_image | current_preimage

            # Check if we have found everything.
            if whole_backward_image_constructor == whole_backward_image:
                break

            # Update containers.
            whole_backward_image = whole_backward_image_constructor

            #bdd.reorder(self.__bdd, self.custom_order)

        return whole_backward_image


    """
    Compute the weak basin of an attractor given by one of its states.

    If 'state' belongs to an attractor, this returns the weak basin of that attractor.
    For an arbitrary state, this returns the set of all states that can reach 'state'
    through backward transitions (backward-reachable states).

    Args:
        state (list[int]): Binary values (0 or 1) representing the network state,
            ordered according to the node list.

    Returns:
        set[str]: A set of strings representing the states in the weak basin of the
            attractor or, more generally, the backward-reachable set of states.
    """
    def compute_weak_basin(self, state: list[int]) -> set[str]:

        whole_backward_image = self.compute_weak_basin_BDD(state)

        state_set = set()
        for models in self.__bdd.pick_iter(whole_backward_image, care_vars=self.node_names):
            # state_str = ''
            # for node_name in self.node_names:
            #     state_str += '1' if models[node_name] else '0'
            # state_set.add(state_str)

            s = []
            for node_name in self.node_names:
                s.append(1 if models[node_name] else 0)
            state_set.add(tuple(s))
        
        # return list(self.__bdd.pick_iter(whole_backward_image))

        return state_set


    """
    Compute a BDD representing the strong basin of an attractor given by one of its states.

    If 'state' belongs to an attractor, this returns the strong basin of that attractor. 
    For a transient state, this returns the set of all states from which every trajectory
    eventually reaches the 'state'.

    Args:
        state (list[int]): Binary values (0 or 1) representing the network state,
            ordered according to the node list.

    Returns:
        BDD: A binary decision diagram representing the strong basin. 
    """
    def compute_strong_basin_BDD(self, state: list[int]) -> Function:
        weak_basin = self.compute_weak_basin_BDD(state)

        strong_basin = self.__bdd.false

        # Compute the strong basin as a fixed-point interation of
        # WB \ ( pre(post(WB)\WB) \cap WB )
        while strong_basin != weak_basin:
            
            if strong_basin != self.__bdd.false:
                weak_basin = strong_basin

            # post(WB)
            aux = self._one_step_image_monolithic_with_direction(weak_basin, direction = 'forward')
            # pre(post(WB)\WB)
            aux = self._one_step_image_monolithic_with_direction(aux & (~weak_basin), direction = 'backward')
            # pre(post(WB)\WB) \cap WB
            aux = aux & (weak_basin)
            
            # WB \ ( pre(post(WB)\WB) \cap WB )
            strong_basin = weak_basin & (~aux)

        return strong_basin

    
    """
    Compute the strong basin of an attractor given by one of its states.

    If 'state' belongs to an attractor, this returns the strong basin of that attractor. 
    For a transient state, this returns the set of all states from which every trajectory
    eventually reaches the 'state'.

    Args:
        state (list[int]): Binary values (0 or 1) representing the network state,
            ordered according to the node list.

    Returns:
        set[str]: A set of strings representing the states in the strong basin.
    """
    def compute_strong_basin(self, state: list[int]) -> set[str]:

        strong_basin = self.compute_strong_basin_BDD(state)

        state_set = set()
        for models in self.__bdd.pick_iter(strong_basin, care_vars=self.node_names):
            # state_str = ''
            # for node_name in self.node_names:
            #     state_str += '1' if models[node_name] else '0'
            # state_set.add(state_str)

            s = []
            for node_name in self.node_names:
                s.append(1 if models[node_name] else 0)
            state_set.add(tuple(s))


        # return list(self.__bdd.pick_iter(whole_backward_image))

        return state_set
    

# === Requires testing before intergation into the library ===

    def find_all_attractors_with_ec(self, all_space_constraints: Function = None,
                            path_to_file: str = None,
                            filename: str = None,
                            verbose: bool = False,
                            environmental_condition: dict[str, bool] = {}) -> list[Function]:
        
        ec = None
        if len(environmental_condition) > 0:
            input_node_names = set(self.getInputNodeNames())

            for key in environmental_condition.keys():
                if key not in input_node_names:
                    raise ValueError(f"Wrong specification of the environmental condition: {key} is not an input node of the model!")

        start_time = time.time()
        sys.setrecursionlimit(self._RECURSION_LIMIT)

        if verbose:
            print("Finding all attractors", flush=True)
            print(f"The network has {self.number_of_blocks}", flush=True)

        # Prepare containers for the further usage.
        q = deque(
            [
                block_nr
                for block_nr, (_, control_nodes) in self.blocks.items()
                if control_nodes == set()
            ]
        )
        processed_blocks_numbers = set()
        all_variables = set()
        all_attractors = []  # Will have the transit matrix T_loc, and attractor A.

        while q:
            block_nr = q.popleft()

            if block_nr in processed_blocks_numbers:
                continue

            # Find all children of the given block.
            all_control_nodes = self.blocks[block_nr][self._GET_CONTROL_NODES]
            all_parents = set([
                self.node_block_number[node_name] for node_name in all_control_nodes
            ])
            block_variables = self.blocks[block_nr][self._GET_SCC]

            # To correctly find update functions one has to
            # sort variables with respect to the general order.
            correct_order_of_block_variables = []
            for gen in self.node_names:
                if gen in block_variables:
                    correct_order_of_block_variables.append(gen)

            all_block_children = self.block_children[block_nr]
            block_nodes_next = [
                var + "_next" for var in correct_order_of_block_variables
            ]
            block_update_functions_bdd = [
                self.node_name_to_bdd_update_function[node_name] for node_name in correct_order_of_block_variables
            ]

            # All parent blocks are already processed.
            if all_parents.issubset(processed_blocks_numbers):

                # If the block is elementary block.
                if self.__is_elementary_block(all_parents):
                    if verbose:
                        print(f"The elementary block nr. {block_nr} is processed")

                    # Find transition matrix of the block.
                    T_loc = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                   nodes_next=block_nodes_next,
                                                                   functions=block_update_functions_bdd)
                    
                    # With this transition matrix find all attractors of the elementary block.
                    block_environmental_condition = dict()
                    for block_var in block_variables:
                        if block_var in environmental_condition:
                            block_environmental_condition[block_var] = environmental_condition[block_var]
                    ec = self.__bdd_representation_of_boolean_state(block_environmental_condition)
                    block_attractors = self.__find_all_attractors_in_scc_block(T_loc,
                                                                               correct_order_of_block_variables,
                                                                               all_space=ec)

                    # Mark the block as already processed.
                    processed_blocks_numbers.add(block_nr)

                    # If that is the first elementary block then just save the result:
                    # i.e. save transition matrix and the attractor states.
                    if not all_attractors:
                        all_attractors = [(T_loc, block_attractors)]
                    else:
                        # If the Boolean Network (BN) contains more than one elementary block,
                        # the results from these blocks must be merged.
                        #
                        # The merging procedure is straightforward:
                        # 1. Apply a logical "OR" operation to the transition matrices of the two elementary blocks.
                        #    Before doing so, extend each matrix to include all variables from both blocks —
                        #    this step preserves the asynchrony of the system.
                        # 2. For each attractor A from the previous elementary block,
                        #    combine it with each attractor B from the current elementary block
                        #    by computing their product (A & B).
                        if verbose:
                            print("Merging elementary blocks")
                        T_elementary_1, attractors_elementary_1 = all_attractors[0]
                        T_merged, attractors_merged = self.__merge_elementary_blocks(attractors_elementary_1,
                                                                                     T_elementary_1,
                                                                                     all_variables,
                                                                                     T_loc,
                                                                                     block_attractors,
                                                                                     block_variables)

                        # Save merged result to the general container.
                        all_attractors = [(T_merged, attractors_merged)]

                    # Mark block variables as already processed.
                    all_variables |= block_variables
                else:  # The block is not elementary.
                    if verbose:
                        print(f"The non-elementary block nr {block_nr} is processed")

                    # Create transition matrix for non-elementary block. At this stage it does not matter
                    # the dynamic of its control nodes.
                    T_child = self.__create_bdd_from_boolean_network(nodes=correct_order_of_block_variables,
                                                                     nodes_next=block_nodes_next,
                                                                     functions=block_update_functions_bdd)

                    # Extend transition matrix of the block to all variables. Similarly extend transition matrix
                    # of parent block to the block variables.
                    extend_conditions_for_new_block = self.__create_non_movers_conditions_for_variables_set(
                        all_variables)
                    extend_conditions_for_parent_block = self.__create_non_movers_conditions_for_variables_set(
                        block_variables)

                    # Each R_i represents the non-mover conditions for current block.
                    # More precisely by adding R_i we say that: variables from the parent block
                    # has to be fixed - it then preserves asynchronuous update mode.
                    for R_i in extend_conditions_for_new_block:
                        T_child &= R_i

                    # Find all new attractors with constructing realisations of the block with respect
                    # to the parent attractor and its dynamic.
                    new_all_attractors = []

                    # Iterate all attractors and theirs dynamics.
                    for (T_parent, attractor_list_parent) in all_attractors:
                        T_extend_parent = T_parent

                        # Similarly to the above code - we extend transition matrix for the parent
                        # on the child's variables.
                        for R_i in extend_conditions_for_parent_block:
                            T_extend_parent &= R_i

                        # For all attractors found under T_parent transition matrix create an realisation.
                        for parent_attractor in attractor_list_parent:
                            # Cut T_child to the previously found attractor.
                            T_child_extend = T_child & parent_attractor

                            # Find transition graph of the attractor.
                            attractors_dynamic = T_extend_parent & parent_attractor

                            # Create new merged transition matrix.
                            T_new = T_child_extend | attractors_dynamic

                            # Extend all_variables and processed_blocks_number containers.
                            all_variables |= block_variables
                            processed_blocks_numbers.add(block_nr)

                            # Find all attractors of the new merged blocks with the new
                            # transition matrix.
                            new_attractors = self.__find_all_attractors_in_scc_block(T_loc=T_new,
                                                                                     all_block_variables=list(
                                                                                         all_variables),
                                                                                     all_space=parent_attractor)
                            # if verbose:
                            #    print(f"Attractors of the block {block_nr} were found")

                            # Save the result.
                            new_all_attractors.append((T_new, new_attractors))

                    # Update all_attractors - this container keeps now all attractors together with theirs
                    # transition matrices of the new merged block.
                    all_attractors = new_all_attractors.copy()

                # Once the block is processed add all its block children to the queue q.
                for child in all_block_children:
                    q.append(child)

            # The block has parent which is not processed yet. Then put it on the end of
            # the queue.
            else:
                q.append(block_nr)

        # PRINTING STAFF.
        end_time = time.time()
        if verbose:
            print(f"All attractors are found in {int((end_time - start_time) / 60)} minutes")
        counter = 1

        if path_to_file or filename:        
            if not path_to_file:
                path_to_file = os.getcwd()
                print("The path to save results is not given. Results "
                    "are going to be saved in the current directory.\n",
                    flush=True)
            if not filename:
                print("The name of the file is not given. Results are going to be saved to the "
                    "attractors_dict_representation.txt for the dict representation and "
                    "to attractors_list_representation.txt "
                    "for a list representation.", flush=True)
                filename_1 = "attractors_dict_representation.txt"
                filename_2 = "attractors_list_representation.txt"
            else:
                filename_1 = filename + "_dict_representation.txt"
                filename_2 = filename + "_list_representation.txt"

            with open(os.path.join(path_to_file, filename_1), "w") as f, open(os.path.join(path_to_file, filename_2), "w") as g:
                g.write(f"The order is {self.node_names}")
                # all_attractors = [(T, A1), (T2, A2), .... , ], Ti - transition matrix of
                # merged blocks realisation, Ai - the final attractor in this realisation.
                for _, attractor_list in all_attractors:
                    for attractor in attractor_list:
                        f.write(f"Attractor {counter}:\n")
                        g.write(f"Attractor {counter}:\n")
                        f.write("=======================\n")
                        g.write("=======================\n")
                        print(f"Attractor nr {counter}", flush=True)
                        counter += 1
                        print(f"=======================")
                        states_counter = 0
                        # models: {name1 : True, name2 : False ,..., name_n : False}
                        for models in self.__bdd.pick_iter(attractor, care_vars=self.node_names):
                            state_string_dict = "{"
                            state_string_list = "("

                            for node_name in self.node_names:
                                name, value = node_name, int(models[node_name])
                                boolean_state = "1, " if models[node_name] else "0, "
                                state_string_dict += name + " : " + boolean_state
                                state_string_list += boolean_state

                            states_counter += 1
                            state_string_dict = state_string_dict[:-2]
                            state_string_list = state_string_list[:-2]
                            state_string_dict += "}"
                            state_string_list += ")"
                            f.write(state_string_dict + "\n")
                            g.write(state_string_list + "\n")
                            print(state_string_list, flush=True)

                        print(f"Number of states: {states_counter}\n"
                            f"-------------------", flush=True)
                        f.write(f"Number of states {states_counter} \n")
                        g.write(f"Number of states {states_counter} \n")

        # Returning attractors: one has to go through list of tuples and append
        # only bdd's representing attractors.
        remember_atractors = []
        for _, attractor_list in all_attractors:
            for attractor in attractor_list:
                if all_space_constraints:
                    intersection = attractor & all_space_constraints

                    if intersection != self.__bdd.false:
                        remember_atractors.append(intersection)
                else:
                    remember_atractors.append(attractor)

        return remember_atractors
    

    def _verify_trajectory(self, trajectory: list[tuple[int, ], ], verbose: bool = True) -> bool:

        input_node_indexes = [self.node_names.index(el) for el in self.getInputNodeNames()]
        input_node_values = dict()
        for idx in input_node_indexes:
            input_node_values[idx] = trajectory[0][idx]


        for j in range(len(trajectory)-1):

            for idx in input_node_indexes:
                if input_node_values[idx] != trajectory[j+1][idx]:
                    return False

            current_state = trajectory[j]
            next_state = trajectory[j+1]

            if int(sum(abs(np.array(next_state)-np.array(current_state)))) > 1:
                if verbose:
                    print(f"More than one update at time step {j} in the trajectory!")
                    print(f"Current state:\n{current_state}")
                    print(f"Next state:\n{next_state}")
                    self.save_ispl("problematic_bn.ispl")
                return False
            elif int(sum(abs(np.array(next_state)-np.array(current_state)))) == 0:
                # We do not know which node was selected, let's move forward
                pass
            else:
                state = list(current_state)
            
                substitutions = dict()
                for i, node_value in enumerate(state):
                    substitutions[self.list_of_nodes[i]] = self.__bool_algebra.TRUE if node_value == 1\
                                                                    else self.__bool_algebra.FALSE

                node_index = int(abs(np.array(next_state)-np.array(current_state)).nonzero()[0][0])

                fun = self.functions[node_index]

                new_node_value = 1 if fun.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE else 0
                state[node_index] = new_node_value

                if tuple(state) != next_state:
                    if verbose:
                        print(f"Wrong value in time step {j} of the trajectory for node {self.node_names[node_index]}")
                        print(f"Current state:\n{current_state}")
                        print(f"Next state:\n{next_state}")
                        self.save_ispl("problematic_bn.ispl")
                    return False

        return True


    def filter_target_attractors(self, attractors: dict[str, list[State | str, ]] = None) -> dict[str, list[State | str, ]]:

        if self.target_configuration is None:
            return attractors

        target_conf_nodes = [str(s) for s in self.target_configuration.symbols]

        if attractors is None:
            raise ValueError("Not implemented yet!")

        target_attractors = dict()

        for attr_key in attractors.keys():

            for attr_state in attractors[attr_key]:

                substitutions = dict()
                for tc_node_name in target_conf_nodes:
                    idx = self.node_names.index(tc_node_name)
                    if type(attr_state[idx]) == str:
                        substitutions[self.list_of_nodes[idx]] = self.__bool_algebra.TRUE if attr_state[idx] == '1' else self.__bool_algebra.FALSE
                    else:
                        substitutions[self.list_of_nodes[idx]] = self.__bool_algebra.TRUE if attr_state[idx] == 1 else self.__bool_algebra.FALSE

                t = self.target_configuration.subs(substitutions, simplify=True)
               
                if self.target_configuration.subs(substitutions, simplify=True) == self.__bool_algebra.TRUE:
                    target_attractors[attr_key] = attractors[attr_key]
                    break

        return target_attractors


    def get_edge_indices_for_perturbations(self) -> list[int,]:

        if self.ec_fixed_nodes is None:
            return list(range(len(self.edges_order)))

        blocked_nodes_indexes = [self.node_names.index(node) for node in self.ec_fixed_nodes.keys()]
        free_edges_indexes = []
        for edge_idx, (s,t) in enumerate(self.edges_order):
            if (s != t) or (s not in blocked_nodes_indexes):
                free_edges_indexes.append(edge_idx)

        return free_edges_indexes

