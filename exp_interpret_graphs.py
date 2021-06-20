import sys
import os
import copy
import time
from os import mkdir
from os.path import isdir

import tensorflow as tf

from util.config import args
from util.models import GCN
from util.graphprocessor import YANCFG

from Explainer import ExplainerModule

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------------------
# helper code
# function to extract block info
# from node and save to textfile
# ------------------------------


def save_to_text(filename, data):
    """
    saves text data into a file
    """
    with open(filename, "w") as fp:
        fp.write(data)
    return


def get_all_block_info(graph):
    """
    steps through CFG graph and converts block info into str format
    -----------------
    Args:
    graph (nx.Graph): the CFG
    -----------------
    Returns:
    block_info (dict, key=node_name, values=information): the information for each node
    """
    block_info = dict()
    for node in list(graph.nodes(data=True)):
        node_name = node[0]
        node_dict = node[1]  # the information of the block

        if "temp" in node_name:
            continue
        
        out = ""  # the string output per node
        _start = node_dict['Start']
        _prev = node_dict['Prev']
        _ins = node_dict['Ins']
        _next = node_dict['Next']
    
        out += 'node ' + node_name + ' info:\n'
        out += ' start: ' + str(_start) + '\n'
        out += ' prev: ' + str(_prev) + '\n'
        out += ' instructions: \n'
    
        for i in _ins:
            out += '  ' + str(i).strip('()') + '\n'
            
        out += ' next: ' + str(_next) + '\n'

        # assign the information to block_info
        block_info[node_name] = out
    return block_info

# -----------------------------
# Main function
# -----------------------------


def interpret(step_size, feat, graph, node_mask, class_label, all_nodes, node_ordering, data_loader, model, explainer):
    """
    Main interpretability function: Algorithm 2
    Will call CFGExplainer and obtain the class_probability, subgraphs, and node_order
    -------------------
    Args:
    step_size (int): the step size to prune graph from 100 -> 0 %
    feat (float, [#nodes, #features]): the feature matrix of the graph
    graph (nx.Graph): the networkx graph
    node_mask (int [#nodes]): 0,1 for real nodes in graph (0 is for padded nodes)
    class_label (int): the correct class label
    all_nodes (str, [#nodes]): list of node names
    node_ordering (dict, key=numeric_id, value=node_name): the mapping between node number in adjacency matrix to the actual node
    data_loader (class): util class
    model (tf.model): the GNN classifier
    explainer (tf.model): the CFGEXplainer instance
    ------------------
    Returns:
    block_order (str, [#nodes]): the node names ordered from most important -> least important w.r.t classification
    classification_result (dict, key=graph_size, values=[calss.probability, num.nodes]): the classification result from GNN model
    subgraphs (dict, key=graph_size, value=pruned_graph): the pruned graphs
    time_taken (float): the time taken per graph
    """
    time_taken = 0  # computes the time taken for algorithm (disregard the nx.Graph() computation part
    block_order = []  # list of node names from most important -> least important
    # classification_result = dict()  # dictionary, key = graph size, values = [class_prob (classifier), #nodes]
    subgraphs = dict()  # dictionary, key = graph size, value = nx.subgraph()

    tot_num_nodes = sum(node_mask)
    sorted_nodes = None
    node_step_size = int(float(step_size / 100) * tot_num_nodes)

    tstart = time.time()
    # 1. loop through all graph sizes
    for i in range(100, 0, -step_size):
        num_nodes_to_remove = node_step_size
        if i == 100:
            num_nodes_to_remove = 0  # so we have class probability of original graph
            
        with tf.device('/gpu:0'):
            # 2. convert to tensors
            _adj = data_loader.graph_to_adj(graph)
            _adj = tf.convert_to_tensor(_adj, dtype=tf.float32)
            _mask = tf.convert_to_tensor(node_mask, dtype=tf.float32)
            _feat = tf.convert_to_tensor(feat, dtype=tf.float32)
            
            # 3. get embeddings, class probability from CFG classifier (GNN model)
            _emb = model.getNodeEmb((_feat, _adj), training=False)

            subgraphs[i] = copy.deepcopy(graph)
            
            # 4. get result from CFGExplainer
            _, importance = explainer((_feat, _emb, _adj, _mask), training=False)
            
            # 5. loop through nodes left and assign node score to node
            # this is done to map the numeric node id -> the actual node name
            textra1 = time.time()
            node_scores = dict()
            for node in all_nodes:
                node_id = node_ordering[node]  # get the numeric ID for the node
                node_scores[node] = importance[node_id]
            textra2 = time.time()
            time_taken += -(textra2 - textra1)  # remove the extra mapping time from the algorithm

            # sort the node scores from most important -> least important
            sorted_nodes = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)

            # 6. prune the graph
            for j in range(num_nodes_to_remove):
                node, value = sorted_nodes.pop()  # get least important node

                block_order.append(node)  # add the removed node
                all_nodes.remove(node)  # remove the node from current all nodes
                node_mask[node_ordering[node]] = 0  # update the node mask

                # remove all in-edges
                in_edges = list(graph.in_edges(node))
                for u, v in in_edges:
                    graph.remove_edge(u, v)

                # remove all out-edges
                out_edges = list(graph.out_edges(node))
                for u, v in out_edges:
                    graph.remove_edge(u, v)
    
    # 7. update the last 10% nodes
    # the most important node will be added last
    loop_sz = len(sorted_nodes)
    for j in range(loop_sz):
        node, value = sorted_nodes.pop()
        block_order.append(node)
    
    tend = time.time()
    time_taken += (tend - tstart)
    block_order.reverse()  # reverse the block order, now the nodes are from most important -> least important

    return block_order, subgraphs, time_taken


def scaled_interpret_experiment(malware_name, class_label, graph_load_path, path_mod, model, explainer):
    """
    will iteratively run the interpret() function on each graph
    -------------------
    Args:
    malware_name (str): the namae of the malware
    class_label (int): the class label id of the malware
    graph_load_path (str): the path to load graphs
    model (tf.Model): the classifier tensorflor model
    explainer (tf.explainer) the explaienr model
    """
    tot_time = 0
    # 1. loop through all graph names
    data_loader = YANCFG()  # class to load the graphs
    # info_loader = BlockProcessor()  # class that converts node/block info into text format
    graph_path = graph_load_path + '/' + path_mod + '/' + malware_name + '/0_list.txt'
    names = data_loader.load_graphnames(graph_path)

    print('malware = ', malware_name, 'started')
    for name in names:

        if args.debug:
            print('loading', name)

        load_path = graph_load_path + '/' + path_mod + '/' + malware_name + '/' + name
        graph, feat, _, node_mask, all_nodes = data_loader.parse_yancfg_sample(load_path, class_label, get_adj=False)
        node_ordering = {node: i for i, node in enumerate(graph.nodes(data=False))}

        num_all_nodes = len(all_nodes)
        # 2. obtain the block information, then call the interpret() function
        block_info = get_all_block_info(graph)

        # 3. obtain the results
        step_size = 10
        block_order, subgraphs, time_taken = interpret(step_size, feat, graph, node_mask, class_label, all_nodes, node_ordering, data_loader, model, explainer)

        # 4. save the results: top_blocks.pickle, top_blocks.txt, class_probability.txt, <adjacency matrices of subgraphs>
        graph_name = name.strip('padded_')
        graph_name = graph_name.strip('gpickle')
        save_path = './results/interpretability_results/' + malware_name + '/' + graph_name
        if isdir(save_path) is False:
            mkdir(save_path)

        # 4.1. save the block order in pickle
        graph_name = name.strip('.gpickle').strip('padded_')
        filename = save_path + '/top_blocks.pickle'
        data_loader.save_pickle(filename, block_order)

        # 4.2. save top blocks in text format
        output_str = "CFGExplainer result: [malware = " + malware_name + " |  graph = " + graph_name + "| #nodes = " + str(num_all_nodes) + "]\n\n"
        for i, node in enumerate(block_order):
            output_str += "node :" + str(i + 1) + "\n"
            output_str += block_info[node]
            output_str += "\n"

        filename = save_path + '/results_top_blocks.txt'
        save_to_text(filename, output_str)

        # 4.3. save 10% and 20% graph

        filename = save_path + "/subgraph_10percent.gpickle"
        data_loader.save_graph(filename, subgraphs[10])

        filename = save_path + "/subgraph_20percent.gpickle"
        data_loader.save_graph(filename, subgraphs[20])

        print('+ done: ', graph_name, ' | time: ', time_taken)
        tot_time += time_taken

    tot_time /= len(names)
    print('malware ', malware_name, 'completed | time = ', tot_time, '\n')
    return

# -----------------------------
# running main function
# -----------------------------


def main(arguments):
    """
    Main function
    ----------------
    Args:
    arguments: the sys.args for running code
    """

    # other arguments are left intact as defaults, check config.py
    # add new arguments: model
    args.d = 13  # the number of features (fixed)
    args.c = 12  # the number of classes (fixed)
    args.n = 4690  # the number of nodes, fixed for experiment
    args.batch_size = 1  # batch size
    args.path = str(arguments[0])  # the path to load the data
    args.hiddens = str(arguments[1])  # '1024-512-128'
    args.model_name_flag = str(arguments[2])  # 'trial_gcn_'
    args.save_path = './checkpoints/' + args.model_name_flag
    args.dataset = str(arguments[3])  # 'yancfg_test'
    explainer_name = str(arguments[4])  # sample1_ep300_b32_elr0001_
    args.embnormlize = False  # keep this False: else the output becomes NaN

    # add arguments: for logging results
    args.disable_tqdm = True  # make False to see progress bar
    
    # debugging argument
    args.debug = False  # prints out the data loading step + loads only 1 graph per sample
    if args.debug:
        print("Experimenting in DEBUG mode!")

    print('> Dynamic Prune graphs [will run CFGExplainer iteratively]')
    # running the code
    
    malware_list = {
        'Bagle': 0,
        'Benign': 1,
        'Bifrose': 2,
        'Hupigon': 3,
        'Ldpinch': 4,
        'Lmir': 5,
        'Rbot': 6,
        'Sdbot': 7,
        'Swizzor': 8,
        'Vundo': 9,
        'Zbot': 10,
        'Zlob': 11
    }
    
    print('\n arguments: \n', args)
    # loading the model
    model = GCN(input_dim=args.d, output_dim=args.c)
    model.load_weights(args.save_path + args.dataset)  # load the weights
    print('+ loaded surrogate model: ', model)

    # loading the explainer
    # explainer_name = "_ep300_b32_elr00005_sciflow_CFGExplainer_"  # MLP-version1
    args.explainer_path = './checkpoints/explainer_' + explainer_name + args.model_name_flag + args.dataset  # path to save the explainer model
    explainer = None
    device = '/gpu:0'  # or '/cpu:0' or CPU
    with tf.device(device):
        explainer = ExplainerModule(model=model, output_dim=args.c)
        explainer.load_weights(args.explainer_path)
        print('+ loaded explainer model: ', explainer)

    for malware_name, class_label in malware_list.items():
        save_path = "./results/interpretability_results/" + malware_name
        if isdir(save_path) is False:
            mkdir(save_path)
        print('\n>> running ', malware_name, ' CFGExplainer experiment')
        scaled_interpret_experiment(malware_name, class_label, args.path, 'padded_train', model, explainer)
    
    return


# running the code
if __name__ == "__main__":
    print("sys.args: ", sys.argv)
    main(sys.argv[1:])
