import pickle
import networkx as nx
import tensorflow as tf


class YANCFG:

    """
    yancfg class for preprocessing the dataset
    """

    def init(self):
        return

    def load_graphnames(self, filename):
        """
        loads the 0_list.txt to read the graph names
        --------------
        Args
        filename: the abs.path to 0_list.txt
        --------------
        Returns
        data (list[str]): the names of all *.gpickles to load
        """
        data = []
        # path = filename.strip('0_list.txt')
        with open(filename, 'r') as f:
            for line in f.readlines():
                if '.gpickle' in line:
                    data.append(line.strip())
        return data

    def parse_yancfg_sample(self, filename, label, num_classes=None, get_adj=True):
        """
        loads the yancfg graph, feature vectors
        ---------------
        Args
        filename (str): the filename of graph to load
        label (int): the label-ID
        num_classes (int): number of malware classes
        ---------------
        Returns
        graph (nx.Graph): graph with n-nodes
        feat (numpy [n,d]): feature vectors for all n-nodes
        one_hot_label (list [c]): one-hot encoded label
        mask (list [n]): one-hot mask for correct nodes
        actual_nodes (list [str]): the list of nodes
        """
        # print(filename)
        graph = nx.read_gpickle(filename)
        
        # feature vector
        feat = list(nx.get_node_attributes(graph, "feat").values())

        # label vector (one hot encoded)
        num_classes = 12
        one_hot_label = [0 for _ in range(num_classes)]
        one_hot_label[label] = 1
        
        # getting the node masks
        # mask = [0 for _ in range(len(graph.nodes))]
        mask = []
        actual_nodes = []
        for i, node in enumerate(graph.nodes):
            # print(node)
            if "temp" in node:
                mask.append(0)
            else:
                mask.append(1)
                actual_nodes.append(node)

        if get_adj:
            graph = nx.to_numpy_matrix(graph).tolist()  # obtains the adjacency matrix
        
        # print(mask)
        return graph, feat, one_hot_label, mask, actual_nodes

    def load_yancfg_data(self, path, path_mod, malware_list, debug=False):
        """
        loads a complete yancfg dataset
        modification: extracts also the graph name and type
        ---------------------
        Args
        path (str): abs.path to load graphs
        path_mod (str): string modifier for dataset with padded_*
        malware_list (dict): a dicitonary for malware type: malware label
        debug (bool): True if in debug mode
        ---------------------
        Returns
        dataset (tf.data.Dataset): tensorflow dataset (graphs, features, labels, ids)
        info (dict): key=ids, values=(sample_name, malware_type)
        num_samples (int): total number of samples loaded
        """
        graphs, feats, labels, ids = [], [], [], []
        node_masks = []
        sample_id = 0
        info = {}  # modification: dict() to store malware_name and type
        for malware_name, label in malware_list.items():
            
            graph_path = path + '/' + path_mod + '/' + malware_name + '/0_list.txt'
            names = self.load_graphnames(graph_path)
            
            num_samples = 0
            for name in names:
                # keeping track of the information
                info[sample_id] = (name, malware_name)
                ids.append(sample_id)
                sample_id += 1
                # loading the graph
                if debug:
                    print('loading ', name)
                load_path = path + '/' + path_mod + '/' + malware_name + '/' + name
                graph, feat, one_hot_label, node_mask, _ = self.parse_yancfg_sample(load_path, label, 13)  # change to 13
                graphs.append(graph)
                feats.append(feat)
                labels.append(one_hot_label)
                node_masks.append(node_mask)
            
                # in debug mode only load 1 graph per type 
                num_samples += 1
                if debug:
                    if num_samples >= 1:
                        break
            print('+ loaded', path_mod, malware_name)

        # convert to tensors
        graphs = tf.convert_to_tensor(graphs, dtype=tf.float32)
        feats = tf.convert_to_tensor(feats, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        ids = tf.convert_to_tensor(ids, dtype=tf.int64)  # the keys to info {} dict
        nmask = tf.convert_to_tensor(node_masks, dtype=tf.float32)  # node masks, must be of same type 
        # create datasets
        graphs = tf.data.Dataset.from_tensor_slices(graphs)
        feats = tf.data.Dataset.from_tensor_slices(feats)
        labels = tf.data.Dataset.from_tensor_slices(labels)
        ids = tf.data.Dataset.from_tensor_slices(ids)
        nmask = tf.data.Dataset.from_tensor_slices(nmask)
    
        # zip to create dataset
        dataset = tf.data.Dataset.zip((graphs, feats, labels, ids, nmask))
        
        return dataset, info, num_samples

    def save_pickle(self, filename, data):
        """
        saves data into pickle
        """
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
        return

    def save_graph(self, filename, graph):
        """
        saves nx graph into pickle
        """
        nx.write_gpickle(graph, filename)
        return

    def graph_to_adj(self, graph):
        """
        converts a graph to a numpy adjacency list[list]
        """
        return nx.to_numpy_matrix(graph).tolist()

