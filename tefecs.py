import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
import math
from itertools import combinations
from collections import defaultdict
import subprocess
from tqdm import tqdm
import numpy as np
import networkx as nx
import pickle

import io
from contextlib import redirect_stdout

from boolnetlab import BN_Realisation as BN # The core class for Boolean network analysis
from boolnetlab import state2bin, int2bin, bin2state, is_GPU_available, parse_cabean_attractors

# ==== Settings ====

NUM_REPETITIONS = 20
MAX_PERTURBATIONS = 30
K = 5

NUM_TEST_REPETITIONS = 20

AUX_MODEL_FOLDER = os.path.join("tefecs_experiments")

CABEAN_PATH = os.path.join(Path.home(), "cabean")

ATTRACTOR_TIMEOUT = 60

CHECK_CODE_CORRECTNESS = True

# ==================

def match(state: str, state_template: str):

    assert len(state) == len(state_template)

    for i in range(len(state)):
        if state_template[i] != '-':
            if state[i] != state_template[i]:
                return False

    return True


def sample_one_edge_per_node(bn, k: int) -> list[int]:
    perturbation = []
    helper_dict = defaultdict(list)

    for s_node_ind, t_node_ind in bn.edges_order:
        if bn.ec_fixed_nodes is not None:
            fixed_node_indexes = [bn.node_names.index(fixed_node_name) for fixed_node_name in bn.ec_fixed_nodes.keys()]
            if (s_node_ind != t_node_ind) or (s_node_ind not in fixed_node_indexes):
                helper_dict[t_node_ind].append(s_node_ind)
        else:
            helper_dict[t_node_ind].append(s_node_ind)
    
    for t_node_ind in random.sample(list(helper_dict.keys()), k=k):
        s_node_ind = random.sample(helper_dict[t_node_ind], k=1)[0]
        perturbation.append(bn.edges_order.index((s_node_ind,t_node_ind)))

    return sorted(perturbation)


def train(bn, 
          log_folder: str,
          log_filename: str,
          attractor_states: dict[str, set] | None = None,
          cabean: bool = False,
          one_edge_per_node: bool = False,
          verbose: bool = False):

    if attractor_states is None:
        bn_attractors = bn.find_all_attractors()
        attractor_states = bn._enumerate_attractor_states(bn_attractors)
        cabean = False

    if verbose:
        print(attractor_states)

    print(f"Training phase ...")

    if len(attractor_states.keys()) == 1:
        print("Single attractor detected! Nothing to do. Exiting...")
        # sys.exit()
        return None
    
    G = nx.DiGraph()

    for source_attractor_key in tqdm(attractor_states.keys()):
    
        if verbose:
            print(f"Processing source attractor {source_attractor_key} ...")
        
        source_attractor = attractor_states[source_attractor_key]
        
        reachability_dict = dict()

        # Block perturbing input nodes that are fixed by the environmental conditions setting.
        valid_edges_for_perturbation = bn.get_edge_indices_for_perturbations()
        blocked_edges_indexes = set(range(bn.getNumEdges())).difference(set(valid_edges_for_perturbation))
        edges = bn.get_edges_order()
        blocked_edges = [edges[idx] for idx in blocked_edges_indexes]
        print(f"Edges excluded from perturbations: {blocked_edges}")

        for k in range(1,K+1):

            num_non_input_nodes = bn.num_nodes -  len(bn.getInputNodeNames())

            max_comb = math.comb(num_non_input_nodes, k)

            if MAX_PERTURBATIONS > int(max_comb / 2):
                print(f"WARNING: Number of non-input nodes is {num_non_input_nodes}. Reducing MAX_PERTURBATIONS to {int(max_comb / 2)} for k = {k}.")
        
            analysed_perturbations = set()

            for _ in range(min(MAX_PERTURBATIONS, int(max_comb / 2))):

                if not one_edge_per_node:
                    # perturbation = sorted(random.sample(range(len(bn.edges_order)), k=k))
                    perturbation = sorted(random.sample(valid_edges_for_perturbation, k=k))
                else:
                    perturbation = sample_one_edge_per_node(bn, k=k)
                while tuple(perturbation) in analysed_perturbations:
                    if not one_edge_per_node:
                        # perturbation = sorted(random.sample(range(len(bn.edges_order)), k=k))
                        perturbation = sorted(random.sample(valid_edges_for_perturbation, k=k))
                    else:
                        perturbation = sample_one_edge_per_node(bn, k=k)

                analysed_perturbations.add(tuple(perturbation))


                if CHECK_CODE_CORRECTNESS:
                    for ind in perturbation:
                        if ind not in valid_edges_for_perturbation:
                            print(f"Valid perturbations: {valid_edges_for_perturbation}")
                            print(f"{ind}: {bn.edges_order[ind]}")
                            raise ValueError(f"Wrong perturbation: {perturbation}")

            
                attractor_reachability_counts = dict()
                for attr_key in attractor_states.keys():
                    attractor_reachability_counts[attr_key] = 0
            
                for _ in range(NUM_REPETITIONS):
                    
                    bn.remove_edges(perturbation, verbose=False)
                    if not cabean:
                        initial_attractor_state = random.choice(tuple(source_attractor))
                    else:
                        initial_attractor_state_str = random.choice(tuple(source_attractor))
                        initial_attractor_state = []
                        for bit in initial_attractor_state_str:
                            if bit == '0':
                                initial_attractor_state.append(0)
                            elif bit == '1':
                                initial_attractor_state.append(1)
                            elif bit == '-':
                                initial_attractor_state.append(random.randint(0,1))
                            else:
                                raise ValueError(f"Unknown bit {bit}. Expected 0, 1, or -.")
                        initial_attractor_state = tuple(initial_attractor_state)
                    # Redirect stdout to an empty StringIO buffer to supress the printing of the "CUDA is not available ..." warning
                    with redirect_stdout(io.StringIO()):
                        state = bn.simulate(nsteps=10000, init_state=initial_attractor_state, full_trajectory=False)[0]
                    bn.restore_original_BN()
            
                    is_attractor_state = False
                    while not is_attractor_state:
                        state = bn.simulate_asynchronous(state, num_steps=1)

                        if CHECK_CODE_CORRECTNESS:
                            for nn in bn.ec_fixed_nodes.keys():
                                node_idx = bn.node_names.index(nn)
                                v = True if state[node_idx] == 1 else False
                                try:
                                    assert v == bn.ec_fixed_nodes[nn]
                                except:
                                    print(v)
                                    print(bn.ec_fixed_nodes[nn])
                                    print(initial_attractor_state[node_idx])
                                    print(state)
                                    print(node_idx)
                                    print(bn.ec_fixed_nodes.keys())
                                    print([bn.node_names.index(nn) for nn in bn.ec_fixed_nodes.keys()])
                                    raise ValueError

                        for attr_key in attractor_states.keys():
                            if not cabean:
                                if tuple(state) in attractor_states[attr_key]:
                                    attractor_reachability_counts[attr_key] += 1
                                    is_attractor_state = True
                                    break
                            else:
                                state_str = state2bin(tuple(state))
                                for attr_state_template in attractor_states[attr_key]:
                                    if match(state_str, attr_state_template):
                                        attractor_reachability_counts[attr_key] += 1
                                        is_attractor_state = True
                                        break
                                if is_attractor_state:
                                    break

                reachability_dict[tuple(perturbation)] = attractor_reachability_counts
        
        if verbose:
            print(reachability_dict)
        
        # Select minimal perturbations amongst the most reliable ones.
        
        best_reliability = defaultdict(int)
        optimal_perturbations = defaultdict(list)
        
        for p_key in reachability_dict.keys():
            for attr_key in reachability_dict[p_key].keys():
                if (attr_key != source_attractor_key) and (reachability_dict[p_key][attr_key] != 0):
                    if best_reliability[attr_key] == reachability_dict[p_key][attr_key]:
                        optimal_perturbations[attr_key].append(p_key)
                    elif best_reliability[attr_key] < reachability_dict[p_key][attr_key]:
                        best_reliability[attr_key] = reachability_dict[p_key][attr_key]
                        optimal_perturbations[attr_key] = [p_key]
                    else:
                        pass
        
        if verbose:
            print(best_reliability)
            print(optimal_perturbations)
        
        # For each target attractor, select one of the shortest perturbations
        for attr_key in optimal_perturbations:
            shortest = K + 1
            optimal_perturbation = None
            for p in optimal_perturbations[attr_key]:
                if len(p) < shortest:
                    shortest = len(p)
                    optimal_perturbation = p
     
            G.add_edge(source_attractor_key, attr_key, weight=shortest, perturbation=optimal_perturbation, label=optimal_perturbation)

    with open(os.path.join(log_folder, f"{log_filename}_control_graph.pkl"), 'wb') as file:
        pickle.dump([G, bn.get_edges_order()], file)

    # Draw the control graph
    control_graph_filename = os.path.join(log_folder, f"{log_filename}_control_graph.pdf")

    graph = nx.drawing.nx_agraph.to_agraph(G)
    graph.layout('dot')
    # graph is an instant of the PyGraphviz.AGraph class
    graph.draw(control_graph_filename)

    return G


def predict(bn, G, source_attractor_key: str, target_attractor_key: str, attractor_states, cabean = False, num_tests: int = 10, verbose: bool = False):

    NUM_TESTS = num_tests

    try:
        optimal_control_strategy = nx.shortest_path(G, source=source_attractor_key, target=target_attractor_key, weight="weight")
        print(f"The optimal control strategy in accordance with the control graph is: {optimal_control_strategy} of length {len(optimal_control_strategy)-1}")
    except nx.NetworkXNoPath:
        print(f"No path from {source_attractor_key} to {target_attractor_key} was found in the control graph!")
        return None, None
    
    assert source_attractor_key != target_attractor_key
    
    trajectory_lengths = []
    
    for i in tqdm(range(NUM_TESTS)):
    
        trajectory_length = 0
    
        current_attractor_key = source_attractor_key
        
        target_attractor_reached = False
        while not target_attractor_reached:
    
            try:
                control_strategy = nx.shortest_path(G, source=current_attractor_key, target=target_attractor_key, weight="weight")
            except nx.NetworkXNoPath:
                if verbose:
                    print(f"No path from {source_attractor_key} to {target_attractor_key} was found in the control graph!")
                # Let's restart, i.e. start from the first step of the optimal strategy once more
                current_attractor_key = source_attractor_key
                continue

            cs_current_attr_key = control_strategy[0]
            cs_next_attr_key = control_strategy[1]
            assert cs_current_attr_key == current_attractor_key
    
            perturbation = G.edges[cs_current_attr_key, cs_next_attr_key]['perturbation']
            
            trajectory_length += 1
    
            bn.remove_edges(list(perturbation), verbose=False)
            current_attractor = attractor_states[current_attractor_key]
            if not cabean:
                initial_attractor_state = random.choice(tuple(current_attractor))
            else:
                initial_attractor_state_str = random.choice(tuple(current_attractor))
                initial_attractor_state = []
                for bit in initial_attractor_state_str:
                    if bit == '0':
                        initial_attractor_state.append(0)
                    elif bit == '1':
                        initial_attractor_state.append(1)
                    elif bit == '-':
                        initial_attractor_state.append(random.randint(0,1))
                    else:
                        raise ValueError(f"Unknown bit {bit}. Expected 0, 1, or -.")
                initial_attractor_state = tuple(initial_attractor_state)
            
            # state = bn.simulate_asynchronous(initial_attractor_state, num_steps=10000)
            # Redirect stdout to an empty StringIO buffer to supress the printing of the "CUDA is not available ..." warning
            with redirect_stdout(io.StringIO()):
                state = bn.simulate(nsteps=10000, init_state=initial_attractor_state, full_trajectory=False)[0]
            bn.restore_original_BN()
    
            attractor_reached = False
            while not attractor_reached:
                state = bn.simulate_asynchronous(state, num_steps=1)
                    
                for attr_key in attractor_states.keys():
                    if not cabean:
                        if tuple(state) in attractor_states[attr_key]:
                            current_attractor_key = attr_key
                            attractor_reached = True
                            break
                    else:
                        state_str = state2bin(tuple(state))
                        for attr_state_template in attractor_states[attr_key]:
                            if match(state_str, attr_state_template):
                                current_attractor_key = attr_key
                                attractor_reached = True
                                break
                        if attractor_reached:
                            break
    
            if current_attractor_key == target_attractor_key:
                trajectory_lengths.append(trajectory_length)
                target_attractor_reached = True
    
    if verbose:
        print(f"Trajectory lengths: {trajectory_lengths}")
        print(f"Mean trajectory length: {np.mean(trajectory_lengths)}")
        print(f"Optimal control strategy of length {len(optimal_control_strategy)-1}: {optimal_control_strategy}")

    return trajectory_lengths, optimal_control_strategy


def run_CABEAN_seq(model_file_path: str, verbose: bool=False):

    if verbose:
        print(f"Running CABEAN on {model_file_path} ...")

    cabean_path = CABEAN_PATH
        
    try:
        args = cabean_path + " -compositional 2 " + model_file_path
        output = subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, shell=True, timeout=ATTRACTOR_TIMEOUT)
        return((output, 0))
    except subprocess.CalledProcessError as e:
        print(f"Error occured with code {e.returncode}!")
        partial_output = e.output
        return((partial_output, e.returncode))
    except subprocess.TimeoutExpired as e:
        print(f"Attractor computation timed out after {e.timeout} seconds.")
        partial_output = e.output
        return((partial_output, None))


def computeAttractorsWithCabean(bn, log_folder: str, log_filename: str, verbose: bool=False):

    ### === Run CABEAN for all possible environmental conditions ===
    
    failed_env_conditions = []
    total_num_attractors = 0

    input_nodes = bn.getInputNodeNames()
    num_input_nodes = len(input_nodes)
    if verbose:
        print(f"Input nodes: {input_nodes}")
        print(f"Number of input nodes: {num_input_nodes}")

    # Preparing models
    for env_condition in range(2**num_input_nodes):

        if verbose:
            print(f"Making model file for environmental condition {env_condition} ... ", flush=True)
         
        env_condition_str = '\t'
        if num_input_nodes != 0:
            for input_node, node_value in zip(input_nodes, bin2state(int2bin(env_condition,num_input_nodes))):
                env_condition_str = env_condition_str + 'M.' + input_node
                if node_value == 1:
                    env_condition_str += '=true and '
                else:
                    env_condition_str += '=false and '
            env_condition_str = env_condition_str[:-5]
        else:
            env_condition_str += f"M.{bn.node_names[0]}=false or M.{bn.node_names[0]}=true"
        env_condition_str += ';'
        
        model_file_path = os.path.join(log_folder, f"{log_filename}_env_cond_{env_condition}.ispl")
        bn.save_ispl(model_file_path, env_condition_str, verbose=False)

    # Running CABEAN for each environmental condition
    results = [run_CABEAN_seq(os.path.join(log_folder, f"{log_filename}_env_cond_{env_condition}.ispl")) for env_condition in range(2**num_input_nodes)]

    attractors = dict()
    
    attractor_key_index, attractor = 0, []
    total_cabean_runtime = 0.0
    for env_condition, output_tuple in enumerate(results):

        output = output_tuple[0]
        exit_code = output_tuple[1]

        if verbose:
            print(output, flush=True)
        with open(os.path.join(log_folder, f"cabean_attractors_{env_condition}.txt"), 'w') as file:
            file.write(str(output))

        if exit_code is None:
            # Timeout took place
            failed_env_conditions.append(env_condition)
            if verbose:
                print(f"WARNING: Attractor computation timed out for environmental condition {env_condition}!")
        elif exit_code == 0:
            parts = output.split("number of attractors =")
            num_attractors = int(parts[1].split('\n')[0])
            total_num_attractors += num_attractors
            # print(f"\tNumber of attractors: {num_attractors}", flush=True)

            # Extract individual attractors
            for line in output.split('\n'):
                if len(line) > 2:
                    if "find attractor #" in line:
                        if len(attractor) > 0: # Check whether this is not the first attractor
                            attractors['A' + str(attractor_key_index)] = set(attractor)
                            attractor_key_index += 1
                            attractor = []
                    elif line[0] == '0' or line[0] == '1' or line[0] == '-':
                        i = 0
                        state_template = ''
                        while i < len(line[:-4]):
                            state_template += line[i]
                            i += 2
                        attractor.append(state_template)
                    elif "time for attractor detection" in line:
                        print(f"Cabean runtime: {float(line.split('=')[1].split(' ')[0])}s")
                        total_cabean_runtime += float(line.split('=')[1].split(' ')[0])

            if len(attractor) > 0:
                attractors['A' + str(attractor_key_index)] = set(attractor)
                attractor_key_index += 1
                attractor = [] # Prepare for next attractor or free memory
        else:
            failed_env_conditions.append(env_condition)
            if verbose:
                print(f"Attractor computation failed on model with environmental condition {env_condition}")
                print(f"  Error message: {output}")
                print(f"  Exit code: {exit_code}")


    msg = str(f"Total number of attractors found for all environmental conditions: {total_num_attractors}")
    print(msg, flush=True)
    msg = str(f"Attractor computation failed for the following environmental conditions: {failed_env_conditions}")
    print(msg, flush=True)

    ever_failed = (len(failed_env_conditions) != 0)

    return attractors, ever_failed, total_cabean_runtime


def evaluate(log_folder: str,
             log_filename: str,
             bn_size: int | None = None,
             max_parents: int | None = None,
             model_file: str | None = None,
             cabean_file: str | None = None,
             run_cabean: bool = True,
             one_edge_per_node: bool = False,
             verbose: bool = False):

    with redirect_stdout(io.StringIO()):
        if model_file is None:
            bn = BN.generate_random_bn(num_nodes=bn_size, max_parent_nodes=max_parents, min_parent_nodes=1, allow_self_loops=True, allow_input_nodes=True, mode='asynchronous', verbose=True)
            bn.save_ispl(os.path.join(log_folder, log_filename+".ispl"), verbose=verbose)
        else:
            bn = BN.load_ispl(model_file)


    if cabean_file is not None:
        attractor_states, total_cabean_runtime = parse_cabean_attractors(cabean_file)
    else:
        if not run_cabean:
            bn_attractors = bn.find_all_attractors()
            # attractor_states = bn._enumerate_attractor_states(bn_attractors)
            attractor_states = bn._enumerate_attractors(bn_attractors)
        else:
            attractor_states, ever_failed, total_cabean_runtime = computeAttractorsWithCabean(bn, log_folder, log_filename, verbose=False)
            if ever_failed:
                if max_parents is not None:
                    print(f"Attractor computation failed for BN of size {bn.num_nodes} with max parents set to {max_parents}.")
                else:
                    print(f"Attractor computation failed for BN of size {bn.num_nodes}.")
                return None


    num_attractors = len(attractor_states)

    attractor_sizes = []
    for a_key in attractor_states.keys():
        attractor_sizes.append(sum([2**state_template.count("-") for state_template in attractor_states[a_key]]))
    if verbose:
        print(f"Exact attractor sizes: {attractor_sizes}")

    cabean_results = run_cabean or (cabean_file is not None)

    G = train(bn, log_folder, log_filename, attractor_states, cabean=cabean_results, one_edge_per_node=one_edge_per_node, verbose=verbose)

    performance_results = dict()
    no_strategies = []

    target_keys = attractor_states.keys()
    if bn.target_configuration is not None:
        target_keys = bn.filter_target_attractors(attractors=attractor_states).keys()
        if len(target_keys) == 0:
            print("No target: there are no attractors that comply with the target configuration!")
            print(f"Target configuration: {bn.target_configuration}")
            print(f"Nodes specified in the target configuration: {[str(sym) for sym in bn.target_configuration.symbols]}")
            print(f"Attractors: {attractor_states}")
            if CHECK_CODE_CORRECTNESS:
                for attr_key in attractor_states.keys():
                    for attr_state in attractor_states[attr_key]:
                        values = dict() 
                        for sym in bn.target_configuration.symbols:
                            sym_idx = bn.node_names.index(str(sym))
                            values[str(sym)] = attr_state[sym_idx]
                        print(values)

            return None
        
        print(f"Target attractors: {target_keys}")

    for source_attractor_key in attractor_states.keys():
        for target_attractor_key in attractor_states.keys():

            if source_attractor_key == target_attractor_key:
                continue

            if target_attractor_key not in target_keys:
                continue

            # Check whether by the definition of edge removal it is possible to reach the target from the source
            # ToDo: This works only for attractor states represented as strings (possibly with wildcard '-')
            input_node_indexes = [bn.node_names.index(node) for node in bn.getInputNodeNames()]
            for input_node_idx in input_node_indexes:
                if sum([0 if s[input_node_idx] == '0' else 1 for s in attractor_states[source_attractor_key]]) == 0:
                    # In all source attractor states the input node value is set to 0's
                    if sum([1 if s[input_node_idx] == '1' else 0 for s in attractor_states[target_attractor_key]]) == len(attractor_states[target_attractor_key]):
                        print(f"There is no possible edgetic control for {source_attractor_key} -> {target_attractor_key} \
                            as input node {bn.node_names[input_node_idx]} cannot change its value from 0 to 1.")
                        continue
                else:
                    assert sum([1 if s[input_node_idx] == '1' else 0 for s in attractor_states[source_attractor_key]]) == len(attractor_states[source_attractor_key]),\
                    f"The input node {bn.node_names[input_node_idx]} changes value in an attractor!"


            print(f"Applying control for {source_attractor_key} -> {target_attractor_key}")
            trajectory_lengths, optimal_control_strategy = predict(bn, G, source_attractor_key, target_attractor_key, attractor_states, cabean=True, num_tests=NUM_TEST_REPETITIONS)
            if optimal_control_strategy is not None:
                # distance_from_optimal = np.mean(trajectory_lengths) - (len(optimal_control_strategy) - 1)
                # performance_results[(source_attractor_key, target_attractor_key)] = (distance_from_optimal, distance_from_optimal/(len(optimal_control_strategy) - 1))
                performance_results[f"{source_attractor_key} -c-> {target_attractor_key}"] =  {
                    "trajectory_lengths": trajectory_lengths,
                    "optimal_strategy_lengh": len(optimal_control_strategy) - 1,
                    "optimal_strategy": optimal_control_strategy
                }
            else:
                no_strategies.append(f"{source_attractor_key} -> {target_attractor_key}")

    # print(f"Discrepancies of mean strategy lengths (across {NUM_TEST_REPETITIONS} repetitions) from optimal lengths (absolute values and percentages):\n{performance_results}")
    if verbose:
        print("=== Evaluation results ===")
        print(performance_results)
        print(f"No control strategy found for {len(no_strategies)} cases:\n{no_strategies}\n")

    return (performance_results, no_strategies, num_attractors)


# def evaluate_existing_model(folder: str, model_filename: str, cabean: bool = True, verbose: bool = False):

#     with redirect_stdout(io.StringIO()):
#         bn = BN.load_ispl(os.path.join(folder, model_filename))
    
#     if not cabean:
#         bn_attractors = bn.find_all_attractors()
#         attractor_states = bn._enumerate_attractor_states(bn_attractors)
#     else:
#         attractor_states, ever_failed = computeAttractorsWithCabean(bn, log_folder, log_filename, verbose=False)

#     if ever_failed:
#         print(f"Attractor computation failed for BN of size {bn.num_nodes} with max parents set to {max_parents}.")
#         return None

#     num_attractors = len(attractor_states)

#     if not cabean:
#         attractor_sizes = [len(attractor_states[a_key]) for a_key in attractor_states.keys()]
#     else:
#         attractor_sizes = []
#         for a_key in attractor_states.keys():
#             attractor_sizes.append(sum([2**state_template.count("-") for state_template in attractor_states[a_key]]))
#     if verbose:
#         print(f"Attractor sizes: {attractor_sizes}")

#     G = train(bn, log_folder, log_filename, attractor_states, cabean = cabean, one_edge_per_node = one_edge_per_node)

#     performance_results = dict()
#     no_strategies = []

#     for source_attractor_key, target_attractor_key in list(combinations(attractor_states, 2)):
#         print(f"Applying control for: {source_attractor_key} -> {target_attractor_key}")
#         trajectory_lengths, optimal_control_strategy = predict(bn, G, source_attractor_key, target_attractor_key, attractor_states, cabean=True, num_tests=NUM_TEST_REPETITIONS)
#         if optimal_control_strategy is not None:
#             # distance_from_optimal = np.mean(trajectory_lengths) - (len(optimal_control_strategy) - 1)
#             # performance_results[(source_attractor_key, target_attractor_key)] = (distance_from_optimal, distance_from_optimal/(len(optimal_control_strategy) - 1))
#             performance_results[(target_attractor_key, source_attractor_key)] =  {"trajectory_lengths": trajectory_lengths,
#                                                                                   "optimal_strategy_lengh": len(optimal_control_strategy) - 1,
#                                                                                   "optimal_strategy": optimal_control_strategy}
#         else:
#             no_strategies.append(f"{source_attractor_key} -> {target_attractor_key}")

#         print(f"Applying control for: {target_attractor_key} -> {source_attractor_key}")
#         trajectory_lengths, optimal_control_strategy = predict(bn, G, target_attractor_key, source_attractor_key, attractor_states, cabean=True, num_tests=NUM_TEST_REPETITIONS)
#         if optimal_control_strategy is not None:
#             # distance_from_optimal = np.mean(trajectory_lengths) - (len(optimal_control_strategy) - 1)
#             # performance_results[(target_attractor_key, source_attractor_key)] = (distance_from_optimal, distance_from_optimal/(len(optimal_control_strategy) - 1))
#             performance_results[(target_attractor_key, source_attractor_key)] =  {"trajectory_lengths": trajectory_lengths,
#                                                                                   "optimal_strategy_lengh": len(optimal_control_strategy) - 1,
#                                                                                   "optimal_strategy": optimal_control_strategy}
#         else:
#             no_strategies.append(f"{target_attractor_key} -> {source_attractor_key}")

#     # print(f"Discrepancies of mean strategy lengths (across {NUM_TEST_REPETITIONS} repetitions) from optimal lengths (absolute values and percentages):\n{performance_results}")
#     if verbose:
#         print("=== Evaluation results ===")
#         print(performance_results)
#         print(f"No control strategy found for {len(no_strategies)} cases:\n{no_strategies}\n")

#     return((performance_results, no_strategies, num_attractors))


def evaluate_random_pairs(model_file: str,
                          control_graph_file: str,
                          cabean_file: str,
                          num_random_pairs: int = 5,
                          verbose: bool = False):

    with redirect_stdout(io.StringIO()):
        bn = BN.load_ispl(model_file)

    attractor_states, total_cabean_runtime = parse_cabean_attractors(cabean_file)

    num_attractors = len(attractor_states)

    attractor_sizes = []
    for a_key in attractor_states.keys():
        attractor_sizes.append(sum([2**state_template.count("-") for state_template in attractor_states[a_key]]))
    if verbose:
        print(f"Exact attractor sizes: {attractor_sizes}")

    with open(control_graph_file, "rb") as file:
        loaded_data = pickle.load(file)

    G = loaded_data[0]
    edges = loaded_data[1]

    edges_order = []
    for edge_key in edges.keys():
        edge_str = edges[edge_key]
        s, t = edge_str.split(" -> ")
        edges_order.append((bn.node_names.index(s), bn.node_names.index(t)))

    bn.edges_order = edges_order
    assert bn.get_edges_order() == edges

    performance_results = dict()
    no_strategies = []


    valid_source_target_pairs = []
    for source_attractor_key in attractor_states.keys():
        for target_attractor_key in attractor_states.keys():

            if source_attractor_key == target_attractor_key:
                continue

            # Check whether by the definition of edge removal it is possible to reach the target from the source
            # ToDo: This works only for attractor states represented as strings (possibly with wildcard '-')
            reachable = True
            input_node_indexes = [bn.node_names.index(node) for node in bn.getInputNodeNames()]
            for input_node_idx in input_node_indexes:
                if sum([0 if s[input_node_idx] == '0' else 1 for s in attractor_states[source_attractor_key]]) == 0:
                    # In all source attractor states the input node value is set to 0's
                    if sum([1 if s[input_node_idx] == '1' else 0 for s in attractor_states[target_attractor_key]]) == len(attractor_states[target_attractor_key]):
                        if verbose:
                            print(f"There is no possible edgetic control for {source_attractor_key} -> {target_attractor_key} \
                                as input node {bn.node_names[input_node_idx]} cannot change its value from 0 to 1.")
                        reachable = False
                        break
                else:
                    assert sum([1 if s[input_node_idx] == '1' else 0 for s in attractor_states[source_attractor_key]]) == len(attractor_states[source_attractor_key]),\
                    f"The input node {bn.node_names[input_node_idx]} changes value in an attractor!"

            if reachable and nx.has_path(G, source_attractor_key, target_attractor_key):
                valid_source_target_pairs.append((source_attractor_key, target_attractor_key))

    if verbose:
        print(f"Valid source-target attractor pairs: {valid_source_target_pairs}")

    if num_random_pairs > len(valid_source_target_pairs):
        num_random_pairs = len(valid_source_target_pairs)

    random_attractor_pairs = random.sample(valid_source_target_pairs, num_random_pairs)

    for source_attractor_key, target_attractor_key in random_attractor_pairs:

        print(f"Applying control for {source_attractor_key} -> {target_attractor_key}")
        trajectory_lengths, optimal_control_strategy = predict(bn, G, source_attractor_key, target_attractor_key, attractor_states, cabean=True, num_tests=NUM_TEST_REPETITIONS, verbose=verbose)
        if optimal_control_strategy is not None:
            # distance_from_optimal = np.mean(trajectory_lengths) - (len(optimal_control_strategy) - 1)
            # performance_results[(source_attractor_key, target_attractor_key)] = (distance_from_optimal, distance_from_optimal/(len(optimal_control_strategy) - 1))
            performance_results[f"{source_attractor_key} -c-> {target_attractor_key}"] =  {
                "trajectory_lengths": trajectory_lengths,
                "optimal_strategy_lengh": len(optimal_control_strategy) - 1,
                "optimal_strategy": optimal_control_strategy
            }
        else:
            no_strategies.append(f"{source_attractor_key} -> {target_attractor_key}")

    print("=== Top 5 edges ===")

    edge_histogram = defaultdict(int)
    for e in G.edges(data=True):
        pert = e[2]['perturbation']
        for p in pert:
            edge_histogram[p] += 1

    top_5_items = sorted(edge_histogram.items(), key=lambda item: item[1], reverse=True)[:5]

    print(f"Top 5 edges: {top_5_items}")

    for top_edge_idx, _ in top_5_items:
        print(edges[top_edge_idx])

    # print(f"Discrepancies of mean strategy lengths (across {NUM_TEST_REPETITIONS} repetitions) from optimal lengths (absolute values and percentages):\n{performance_results}")
    if verbose:
        print("=== Evaluation results ===")
        print(performance_results)
        print(f"No control strategy found for {len(no_strategies)} cases:\n{no_strategies}\n")

    print("=== Evaluation results in LaTeX ===")

    for control_key in performance_results.keys():
        r = performance_results[control_key]
        s, t = control_key.split(" -c-> ")
        tl = r["trajectory_lengths"]
        optimal_strategy = r["optimal_strategy"]
        optimal_strategy_str = ''
        for i in range(len(optimal_strategy)-1):
            optimal_strategy_str += f"{optimal_strategy[i]} $\\rightarrow$ "
        optimal_strategy_str += f"{optimal_strategy[-1]}"

        latex_str = f"{s} $\\rightarrow$ {t} & {np.mean(tl):.1f} ($\\pm$ {np.std(tl):.1f}) & {optimal_strategy_str}\\\\"

        print(latex_str)

    for edge in G.edges(data=True):

        pert = edge[2]['perturbation']
        latex_str = f"{pert} & "
        for p in pert:
            latex_str += edges[p]
            latex_str += ", "
        latex_str = latex_str[:-2]
        latex_str += "\\\\"

        latex_str = latex_str.replace('v_', '')
        latex_str = latex_str.replace('_', '\\_')
        latex_str = latex_str.replace('->', '$\\rightarrow$')
        
        print(latex_str)

    latex_str = "Top 5 edges by frequency in the control graph:"
    for top_edge_idx, _ in top_5_items:
        latex_str += f"{edges[top_edge_idx]}, "
    latex_str = latex_str[:-2]
    latex_str = latex_str.replace('->', '$\\rightarrow$')
    print(latex_str)


def evaluate_on_random_BNs(num_BNs: int, BN_sizes: list[int,], max_parents: int):

    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # For naming of experiment folders

    if not os.path.exists(AUX_MODEL_FOLDER):
        os.mkdir(AUX_MODEL_FOLDER)
    os.mkdir(os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}"))

    performance_statistics = dict()
    attractor_computation_failure_counter = defaultdict(int)
    attractor_statistics_max_parents = defaultdict(list)
    attractor_statistics_bn_size = defaultdict(list)

    for bn_size in BN_sizes:

        performance_data = defaultdict(list)

        os.mkdir(os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}", f"size_{bn_size}"))

        for mp in range(2,max_parents+1):

            os.mkdir(os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}", f"size_{bn_size}", f"max_parents_{mp}"))
        
            for i in range(num_BNs):

                print(f"Analysing BN of size {bn_size} with max parents {mp} ({i+1})")

                log_folder = os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}", f"size_{bn_size}", f"max_parents_{mp}", f"{i}")
                os.mkdir(log_folder)

                log_filename_stem = f"bn_{bn_size}_{mp}"

                results = evaluate(log_folder, log_filename_stem, bn_size, mp, cabean_file=None, run_cabean=True, one_edge_per_node=True)

                if results is None:
                    attractor_computation_failure_counter[(bn_size,mp)] += 1
                else:
                    performance_results, no_strategies, num_attractors = results

                    attractor_statistics_bn_size[bn_size].append({"num_attractors" : num_attractors, "no_control_strategies" : len(no_strategies)})
                    attractor_statistics_max_parents[mp].append({"num_attractors" : num_attractors, "no_control_strategies" : len(no_strategies)})

                    for key in performance_results.keys():
                        performance_data["control_strategy_lengths"].extend(performance_results[key]["trajectory_lengths"])
                        optimal_length = performance_results[key]["optimal_strategy_lengh"]
                        performance_data["percentage_discrepancy"].extend([((tl - optimal_length)*((tl - optimal_length) > 0))/optimal_length for tl in performance_results[key]["trajectory_lengths"]])

                print("---------------------------------------------------")

        performance_statistics[bn_size] = {"control_strategy_lengths" : performance_data["control_strategy_lengths"],
                                           "percentage_discrepancy" : performance_data["percentage_discrepancy"],
                                           "mean_control_strategy_length" : np.mean(performance_data["control_strategy_lengths"]),
                                           "std_control_strategy_length" : np.std(performance_data["control_strategy_lengths"]),
                                           "min_control_strategy_length" : np.min(performance_data["control_strategy_lengths"]),
                                           "max_control_strategy_length" : np.max(performance_data["control_strategy_lengths"]),
                                           "mean_percentage_discrepancy" : np.mean(performance_data["percentage_discrepancy"]),
                                           "std_percentage_discrepancy" : np.std(performance_data["percentage_discrepancy"]),
                                           "min_percentage_discrepancy" : np.min(performance_data["percentage_discrepancy"]),
                                           "max_percentage_discrepancy" : np.max(performance_data["percentage_discrepancy"])
                                           }

    print("Saving results ...", end=" ")

    experiment_results_file = os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}", f"results_{experiment_timestamp}.pkl")
    with open(experiment_results_file, 'wb') as file:
        pickle.dump([performance_statistics, attractor_computation_failure_counter, attractor_statistics_bn_size, attractor_statistics_max_parents], file)

    experiment_summary_file = os.path.join(AUX_MODEL_FOLDER, f"experiment_{experiment_timestamp}", f"log_{experiment_timestamp}.txt")
    with open(experiment_summary_file, 'w') as file:

        file.write("----------- Settings -----------\n")
        file.write(f"Edge-perturbation samples per size: {MAX_PERTURBATIONS}\n")
        file.write(f"Number of repetitions per edge_perturbation: {NUM_REPETITIONS}\n")
        file.write(f"Maximum simultaneous edge removals: {K}\n")
        file.write(f"Number of control strategie predictions per attractor pair: {NUM_TEST_REPETITIONS}\n")
        file.write(f"Attractor computation timeout: {ATTRACTOR_TIMEOUT} s\n")
        file.write(f"Number of BNs per size: {num_BNs}\n")
        file.write(f"Maximum number of parent nodes: {max_parents}\n")
        file.write(f"Considered BN sizes: {BN_sizes}\n")

        for bn_size in BN_sizes:
            file.write(f"BN size: {bn_size}\n")
            file.write("------------------------\n")
            file.write(f"Mean control strategy length: {performance_statistics[bn_size]['mean_control_strategy_length']}\n")
            file.write(f"Std of control strategy lengths: {performance_statistics[bn_size]['std_control_strategy_length']}\n")
            file.write(f"Min control strategy length: {performance_statistics[bn_size]['min_control_strategy_length']}\n")
            file.write(f"Max control strategy length: {performance_statistics[bn_size]['max_control_strategy_length']}\n")
            file.write(f"Mean percentage discrepancy: {performance_statistics[bn_size]['mean_percentage_discrepancy']}\n")
            file.write(f"Std of percentage discrepancies: {performance_statistics[bn_size]['std_percentage_discrepancy']}\n")
            file.write(f"Min percentage discrepancy: {performance_statistics[bn_size]['min_percentage_discrepancy']}\n")
            file.write(f"Max percentage discrepancy: {performance_statistics[bn_size]['max_percentage_discrepancy']}\n")
            file.write("========================\n\n")
        
    print("done.")


def evaluate_on_given_model(model_file: str, cabean_file: str | None = None, run_cabean: bool = True, verbose: bool = False):

    filename_stem = Path(model_file).stem

    if not os.path.exists(AUX_MODEL_FOLDER):
        os.mkdir(AUX_MODEL_FOLDER)

    log_folder = os.path.join(AUX_MODEL_FOLDER, f"experiment_{filename_stem}")
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    results = evaluate(log_folder, filename_stem, model_file=model_file, cabean_file=cabean_file, run_cabean=run_cabean, one_edge_per_node=True, verbose=verbose)

    print(results)

    with open(os.path.join(log_folder, f"results_{filename_stem}.pkl"), 'wb') as file:
        pickle.dump(results, file)


def main_old():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, help='ISPL file with the BN model.')
    # parser.add_argument('--log-folder', type=str, help='Folder where auxiliary outputs and results will be saved.')
    # parser.add_argument('--filename-stem', type=str)
    # parser.add_argument('--bn-size', type=int, help='Size of the BN to be generated (optional); used only if --model-file is not specified.')
    # parser.add_argument('--max-parents', type=int)
    parser.add_argument('--cabean-file', type=str, help='Optional Cabean output file with a list of attractors.')
    # parser.add_argument('--no-cabean', action='store_false', dest='run_cabean', help='Disable computing exact attractors using cabean.')
    # parser.add_argument('--num-mc-repetitions', type=int, default=1, help='Number of times the Monte Carlo attractor search is run.')
    # parser.add_argument('--verbose', action='store_true', dest='verbose', help='Enable verbose mode.')

    args = parser.parse_args()

    if not is_GPU_available():
        print("------------------------------------------------------------")
        print("WARNING: GPU not available. Falling back to CPU simulations.")
        print("------------------------------------------------------------")

    if args.model_file is None:

        evaluate_on_random_BNs(num_BNs = 5, BN_sizes = [30, 40, 50, 60, 70], max_parents = 5)

    else:

        evaluate_on_given_model(args.model_file, args.cabean_file, verbose=True)
        

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, help='ISPL file with the BN model.')
    parser.add_argument('--control_graph_file', type=str, help='Pickle file with the saved control graph.')
    # parser.add_argument('--log-folder', type=str, help='Folder where auxiliary outputs and results will be saved.')
    # parser.add_argument('--filename-stem', type=str)
    # parser.add_argument('--bn-size', type=int, help='Size of the BN to be generated (optional); used only if --model-file is not specified.')
    # parser.add_argument('--max-parents', type=int)
    parser.add_argument('--cabean-file', type=str, help='Optional Cabean output file with a list of attractors.')
    parser.add_argument('--no-cabean', action='store_false', dest='run_cabean', help='Disable computing exact attractors using CABEAN.')
    parser.add_argument('--num_random_pairs', type=int, default=1, help='Number of random source-target attractor pairs.')
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='Enable verbose mode.')

    args = parser.parse_args()

    if not is_GPU_available():
        print("------------------------------------------------------------")
        print("WARNING: GPU not available. Falling back to CPU simulations.")
        print("------------------------------------------------------------")

    if (args.model_file is not None) and (args.control_graph_file is not None) and (args.cabean_file is not None):

        evaluate_random_pairs(model_file=args.model_file,
                              control_graph_file=args.control_graph_file,
                              cabean_file=args.cabean_file,
                              num_random_pairs=args.num_random_pairs,
                              verbose=args.verbose)

    elif (args.model_file is not None):

        evaluate_on_given_model(model_file=args.model_file, cabean_file=args.cabean_file, run_cabean=args.run_cabean, verbose=args.verbose)

    else:

        print("Nothing to do. Exiting.")


main()
