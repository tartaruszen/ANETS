_author__ = 'Sorour E. Amiri'

import os
import sys
import subprocess
import numpy as np
import copy
import networkx as nx
import time
import getEigenScore as gES
import math
import multiprocessing


class Running_mode(object):
    """
    This class creates an objects which help us to run some specific part of the code
    """
    def __init__(self, input_data, percent, max_itr, tol, num_cores1, num_cores2):
        self.input_data = input_data
        self.percent = percent
        self.max_itr = max_itr
        self.tol = tol
        self.bignum = 999999
        self.num_cores1 = num_cores1
        self.num_cores2 = num_cores2
        self.parallel_feature = False
        self.features_perP = []
        self.parallel1 = False
        self.parallel2 = False
        if num_cores1 > 1:
            self.parallel1 = True
        if num_cores2 > 1:
            self.parallel2 = True


class directory(object):
    """
    This class creates a directory object.
    """

    def __init__(self):
        self.root = ''
        self.edge_dir_original = ''
        self.edge_dir_modified = ''
        self.new_edge_dir = ''
        self.feature_dir = ''
        self.output_dir = ''
        self.ego_feat_dir = ''
        self.intermediate_results = ''

    def set_att(self, root, edge_dir, feature_dir, output_dir):
        self.root = root
        self.edge_dir = edge_dir
        self.feature_dir = feature_dir
        self.output_dir = output_dir

    def get_dir(self, rmode):
        root = rmode.input_data + '/'
        root = root.replace('//', '/')
        edge_dir = root + 'links.txt'
        feature_dir = root + 'features.txt'
        output_dir = root + str(rmode.percent) + '/'
        self.set_att(root, edge_dir, feature_dir, output_dir)
        self.output_dir = root + str(rmode.percent) + '/'
        self.intermediate_results = self.output_dir + 'intermediate_results/'

        if not os.path.exists(self.intermediate_results):
            os.makedirs(self.intermediate_results)


class Data(object):
    """ This class provide all data and the methods related to it
    """

    def __init__(self, _dir):
        self.dir = _dir
        self.nodes = {}
        self.features = {}
        self.edges = {}
        self.label = {}

    def load_normal_data(self, _dir):
        # print('load_normal_data')
        nodes = {}
        edges = {}
        features = {}
        with open(_dir.new_edge_dir, 'r') as f:
            lines = f.readlines()
        lines.pop(0)
        for line in lines:
            line = line.replace('\n', '')
            item = line.split('\t')
            new_edge = [int(item[0]), int(item[1])]
            try:
                edges[str(new_edge)]
            except KeyError:
                edges[str(new_edge)] = new_edge
                for node in new_edge:
                    try:
                        nodes[node]
                    except KeyError:
                        nodes[node] = node

        with open(_dir.feature_dir, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
        for line in lines:
            item = line.split('\t')
            new_node = int(item.pop(0))
            new_feature = []
            for i in item:
                try:
                    new_feature.append(float(i))
                except ValueError:
                    continue
            try:
                features[new_node]
                # print new_node
            except KeyError:
                features[new_node] = new_feature

        self.features = features
        self.nodes = nodes
        self.label = nodes
        self.edges = edges

    def load_data(self, _dir):
        print('load_data')
        nodes = {}
        edges = {}
        features = {}
        sep = '\t'
        with open(_dir.edge_dir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            item = line.split(sep)
            new_edge = [int(item[0]), int(item[1])]
            if not str(new_edge) in edges:
                edges[str(new_edge)] = new_edge
                for node in new_edge:
                    if not node in nodes:
                        nodes[node] = node
            if not str([new_edge[1], new_edge[0]]) in edges:
                edges[str([new_edge[1], new_edge[0]])] = [new_edge[1], new_edge[0]]

        with open(_dir.feature_dir, 'r') as f:
            lines = f.readlines()
        lines.pop(0)
        for line in lines:
            item = line.split(sep)
            new_node = int(item.pop(0))
            new_feature = []
            for i in item:
                try:
                    new_feature.append(float(i))
                except ValueError:
                    continue
            if new_node not in features:
                features[new_node] = new_feature

        self.features = features
        self.nodes = nodes
        self.label = nodes
        self.edges = edges


class Att_graph(object):
    """ This is a graph where each node has a set of attributes
    """

    def __init__(self, nodes, edges, labels, features, _dir, percent, tol):
        self.features = features
        self.feature_dir = _dir.feature_dir
        self.att_summary = _dir.intermediate_results + 'att_summary.txt'
        self.att_score_dir = _dir.intermediate_results + 'att_score.txt'
        self.coarse_att_output_dir = _dir.intermediate_results + 'att_coarse.txt'
        self.att_map_dir = _dir.intermediate_results + 'att_final_map.txt'
        self.att_time_dir = _dir.intermediate_results + 'att_time.txt'
        self.att_time_per_itr = _dir.intermediate_results + 'att_time_per_itr.txt'
        self.att_time_all_itr = _dir.intermediate_results + 'att_time_all_itr.txt'
        self.g_score = _dir.intermediate_results + 'g_score.txt'
        self.att_g_score = _dir.intermediate_results + 'att_g_score.txt'
        self.att_labels_dir = _dir.intermediate_results + 'att_labels.txt'
        self.delta_feature = {}
        self.delta_lambda_direct = {}
        self.lambda0 = 0
        self.estimate_lambda0 = 0
        self.lambda0_direct = 0
        self.f0 = 0
        self.f0_direct = 0
        self.tol = tol
        self.nodes = nodes
        self.edges = edges
        self.labels = labels
        self.delta_lambda = {}
        self.percent = percent
        self.graph = nx.Graph()
        self.map_dir = _dir.intermediate_results + 'final_map.txt'
        self.time_dir = _dir.intermediate_results + 'time.txt'
        self.coarse_input_dir = _dir.root + 'links.txt'
        self.coarse_input = []
        self.link_score_dir = _dir.intermediate_results + 'score.txt'
        self.lambda0_dir =_dir.intermediate_results + 'lambda0.txt'
        self.coarse_output_dir = _dir.intermediate_results + 'coarse.txt'
        self.generate_delta_lambda()
        self.tmp_score_dir = _dir.intermediate_results + 'tmp_score.txt'
        self.tmp_output_dir = _dir.intermediate_results + 'tmp_coarse.txt'
        self.tmp_map_dir = _dir.intermediate_results + 'tmp_final_map.txt'
        self.tmp_time_dir = _dir.intermediate_results + 'tmp_time.txt'

        for key, e in edges.iteritems():
            self.graph.add_edge(e[0], e[1])
        self.feature_arr = []
        self.calculate_f0()
        self.score_dir = _dir.output_dir + '/summary_scores.txt'

    def create_coarse_edge(self):
        self.coarse_input = []
        with open(self.coarse_input_dir) as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            items = line.split('\t')
            n1 = int(items[0])
            n2 = int(items[1])
            v1 = float(items[2])
            v2 = float(items[3])
            self.coarse_input.append([n1, n2, v1, v2])

    def load_link_score(self):
        link_score = []
        link_score_dict = {}
        with open(self.link_score_dir) as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            items = line.split('\t')
            n1 = int(items[0])
            n2 = int(items[1])
            v = float(items[2])
            link_score.append([n1, n2, v])
            self.estimate_lambda0 += v
            link_score_dict[str([n1, n2])] = v
        with open(self.lambda0_dir) as f:
            self.lambda0 = float(f.readline())
        return link_score, link_score_dict

    def get_link_score(self):
        link_score, lambda0 = gES.main(self.coarse_input, self.link_score_dir, self.lambda0_dir, self.tol)
        self.lambda0 = lambda0[0]

        link_score_dict = {}
        for n1, n2, score in link_score:
            self.estimate_lambda0 += score
            link_score_dict[str([n1, n2])] = score
        return link_score, link_score_dict

    def generate_delta_lambda(self):
        self.create_coarse_edge()
        link_score, link_score_dict = self.get_link_score()
        for line in link_score:
            edge = [line[0], line[1]]
            d_lambda = line[2]
            self.delta_lambda[str(edge)] = d_lambda

    def update_centroid(self, current_centroid, cluster, index, moved_nodes):

        if not index[0] == index[1]:
            features_arr = []
            for node in moved_nodes['2new']:
                features_arr.append(self.features[node])
            features_arr = np.array(features_arr)
            Sum_new = features_arr.sum(0)
            Len_new = len(moved_nodes['2new'])

            features_arr = []
            for node in moved_nodes['2old']:
                features_arr.append(self.features[node])
            features_arr = np.array(features_arr)
            Sum_old = features_arr.sum(0)
            Len_old = len(moved_nodes['2old'])

            Sum = Sum_new - Sum_old
            Len = Len_new - Len_old

            key = index[0] #old cluster
            values = cluster[key]
            current_centroid[key] = (current_centroid[key] * (len(values) + Len) - Sum) / len(values)
            key = index[1] #new cluster
            values = cluster[key]
            current_centroid[key] = (current_centroid[key] * (len(values) - Len) + Sum) / len(values)
        return current_centroid

    def cluster2class(self, cluster):
        current_centroid = {}  # starts from 0
        current_assignment = {}  # starts from 0
        for key, values in cluster.iteritems():
            current_centroid[key] = calculateCentroid(values, self.features)
            for value in values:
                current_assignment[value] = key - 1
        return current_centroid, current_assignment

    def acn(self):
        # print('Att_CN')
        self.create_coarse_edge()
        link_score, link_score_dict = self.get_link_score()
        # read mad_dir
        with open(self.map_dir) as f:
            lines = f.readlines()
        num_cluster = 1
        cluster = {}
        for line in lines:
            items = line.split(':')
            nodes = items[1].split(',')
            for i in nodes:
                try:
                    i = int(i)
                    self.labels[i] = num_cluster
                    try:
                        temp = cluster[num_cluster]
                        temp.append(i)
                        cluster[num_cluster] = temp
                    except KeyError:
                        cluster[num_cluster] = [i]
                except ValueError:
                    continue
            num_cluster += 1

        return cluster, link_score, link_score_dict

    def update_delta_f(self, index1, index2, cluster_delta_f, centroid_list, total_delta_f, size_delta_f, cluster,
                       assignment):
        if True: #not rmode.parallel:
            nodes = cluster[index1 + 1] + cluster[index2 + 1]
            f = paralle_dist_2_centroid([assignment, self.features, centroid_list, nodes, [index1, index2]])
        else:
            nodes = cluster[index1 + 1] + cluster[index2 + 1]
            nodes_list = [[] for i in xrange(rmode.num_cores)]
            ii = 0
            for node in nodes:
                nodes_list[ii].append(node)
                ii = (ii + 1) % rmode.num_cores
            inp = []
            for nodes in nodes_list:
                inp.append([assignment, self.features, centroid_list, nodes, [index1, index2]])

            try:
                out = pool.map(paralle_dist_2_centroid, inp)
            except KeyboardInterrupt:
                'terminate all...'
                pool.terminate()
            f = {}
            for o in out:
                for key in o:
                    try:
                        tmp = f[key]
                        f[key] = tmp + o[key]
                    except KeyError:
                        f[key] = o[key]

        for i in [index1, index2]:
            total_delta_f -= cluster_delta_f[i]
            cluster_delta_f[i] = f[i]
            size_delta_f[i] = len(cluster[i + 1])
            total_delta_f += cluster_delta_f[i]

        return cluster_delta_f, total_delta_f, size_delta_f

    def get_cluster_totla_f(self, current_centroid, current_cluster, nodes_perP):
        if not rmode.parallel1:
            for nodes in nodes_perP:
                cluster_delta_f = get_cluster_delta([self.labels, self.features, current_centroid, current_cluster,
                                                    nodes])
        else:
            inp = []
            for nodes in nodes_perP:
                inp.append([self.labels, self.features, current_centroid, current_cluster, nodes])

            try:
                out = pool.map(get_cluster_delta, inp)
            except KeyboardInterrupt:
                'terminate all...'
                pool.terminate()
            cluster_delta_f = {}
            for o in out:
                for key in o:
                    try:
                        tmp = cluster_delta_f[key]
                        cluster_delta_f[key] = tmp + o[key]
                    except KeyError:
                        cluster_delta_f[key] = o[key]

        total_delta_f = 0
        size_delta_f = {}
        for key, value in cluster_delta_f.iteritems():
            size_delta_f[key] = len(current_cluster[key + 1])
            total_delta_f += value

        return cluster_delta_f, total_delta_f, size_delta_f

    def update_current_parameters(self, best_assignment, best_centroid, index1, index2, current_cluster,
                                  cluster_delta_f, total_delta_f, size_delta_f, current_node2C_dist, old_lambda_dict
                                  ,nodes_perP, pure_delta_f, moved_nodes, next_best_cluster):
        # print 'update_current_parameters'
        current_assignment = best_assignment.copy()
        current_centroid = best_centroid
        # cluster_index = [index1, index2]
        current_centroid = self.update_centroid(current_centroid, current_cluster, [index1 + 1, index2 + 1], moved_nodes)

        current_cluster_delta_f, current_total_delta_f, current_size_delta_f = self.update_delta_f(index1, index2,
                                                                                                   cluster_delta_f,
                                                                                                   current_centroid,
                                                                                                   total_delta_f,
                                                                                                   size_delta_f,
                                                                                                   current_cluster,
                                                                                                   best_assignment)

        changed_node_list = {}
        if not index1 == index2:
            for node in current_cluster[index1 + 1]:
                changed_node_list[node] = node
            for node in current_cluster[index2 + 1]:
                changed_node_list[node] = node

        print 'p_ucp'
        if not rmode.parallel2:
            current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster = p_ucp([self, self.nodes, current_cluster,
                                                                         changed_node_list, current_centroid,
                                                                         index1, index2,current_cluster_delta_f,
                                                                         current_node2C_dist, old_lambda_dict,
                                                                         current_assignment,pure_delta_f, next_best_cluster])
        else:
            inp = []
            for nodes in nodes_perP:
                inp.append([self, nodes, current_cluster, changed_node_list, current_centroid, index1, index2,
                            current_cluster_delta_f, current_node2C_dist, old_lambda_dict, current_assignment,
                            pure_delta_f, next_best_cluster])

            try:
                out = pool.map(p_ucp, inp)
            except KeyboardInterrupt:
                'terminate all...'
                pool.terminate()

            v1, v2, v3, v4 = zip(*out)
            Len = len(nodes_perP)
            for i in range(Len):
                for key in nodes_perP[i]:
                    current_node2C_dist[key] = v1[i][key]
                    old_lambda_dict[key] = v2[i][key]
                    pure_delta_f[key] = v3[i][key]
                    next_best_cluster[key] = v4[i][key]

        return current_centroid, current_cluster, current_cluster_delta_f, current_total_delta_f, \
               current_size_delta_f, current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster

    def first_current_parameters(self, best_cluster, best_assignment, best_centroid, current_cluster,
                                 current_node2C_dist, old_lambda_dict, nodes_perP, pure_delta_f, next_best_cluster):
        print 'first_current_parameters'
        s_f = time.time()
        # current_centroid, current_assignment = self.cluster2class(best_cluster)
        current_centroid = deepcopy_cluster(best_centroid)
        current_assignment = copy.deepcopy(best_assignment)
        current_cluster_delta_f, current_total_delta_f, current_size_delta_f = \
            self.get_cluster_totla_f(current_centroid, current_cluster, nodes_perP)
        ############## Det distance of each node to each cluster
        print 'p_fcp'
        if not rmode.parallel1:
            current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster = p_fcp(
                [self, self.nodes, current_cluster, current_centroid, current_cluster_delta_f,
                 current_assignment, pure_delta_f])
        else:
            inp = []
            for nodes in nodes_perP:
                inp.append([self, nodes, current_cluster, current_centroid, current_cluster_delta_f,
                            current_assignment, pure_delta_f])

            try:
                out = pool.map(p_fcp, inp)
            except KeyboardInterrupt:
                'terminate all...'
                pool.terminate()

            v1, v2, v3, v4 = zip(*out)
            Len = len(nodes_perP)
            for i in range(Len):
                for key in nodes_perP[i]:
                    current_node2C_dist[key] = v1[i][key]
                    old_lambda_dict[key] = v2[i][key]
                    pure_delta_f[key] = v3[i][key]
                    next_best_cluster[key] = v4[i][key]
        e_f = time.time()
        # print('first_current_parameters = ' + str(e_f - s_f) + 's')
        return current_centroid, current_cluster, current_cluster_delta_f, current_total_delta_f, \
               current_size_delta_f, current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f,\
               next_best_cluster

    def get_current_parameters(self, best_score, best_delta_lambda, best_delta_f, best_assignment, best_centroid,
                               best_cluster, cluster_delta_f, total_delta_f, size_delta_f, index1, index2,
                               current_node2C_dist, old_lambda_dict, nodes_perP, pure_delta_f, moved_nodes, next_best_cluster):
        current_score = best_score
        current_delta_lambda = best_delta_lambda
        current_delta_f = best_delta_f
        # current_cluster = copy.deepcopy(best_cluster)
        current_cluster = deepcopy_cluster(best_cluster)
        if len(cluster_delta_f) > 0:
            current_centroid, current_cluster, current_cluster_delta_f, current_total_delta_f, current_size_delta_f, \
            current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster = \
                self.update_current_parameters(best_assignment, best_centroid, index1, index2, current_cluster,
                                               cluster_delta_f, total_delta_f, size_delta_f, current_node2C_dist,
                                               old_lambda_dict, nodes_perP, pure_delta_f, moved_nodes, next_best_cluster)

        else:
            current_centroid, current_cluster, current_cluster_delta_f, current_total_delta_f, \
            current_size_delta_f, current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster = \
                self.first_current_parameters(best_cluster, best_assignment, best_centroid, current_cluster, current_node2C_dist,
                                              old_lambda_dict, nodes_perP, pure_delta_f, next_best_cluster)

        return current_score, current_delta_lambda, current_delta_f, current_centroid, \
               current_cluster, current_cluster_delta_f, current_total_delta_f, current_size_delta_f, \
               current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster

    def get_best_parameters(self, score, delta_lambda, delta_f, assignment, centroid, cluster):
        best_score = score
        best_delta_lambda = delta_lambda

        best_delta_f = delta_f
        if len(assignment) > 0:
            best_assignment = assignment.copy()
            best_centroid = centroid
            # best_cluster = copy.deepcopy(cluster)
            best_cluster = deepcopy_cluster(cluster)
        else:
            best_centroid, best_assignment = self.cluster2class(cluster)
            # best_cluster = copy.deepcopy(cluster)
            best_cluster = deepcopy_cluster(cluster)
        return best_score, best_delta_lambda, best_delta_f, best_assignment, best_centroid, best_cluster

    def anets(self, cluster, link_score, link_score_dict, current_score, current_delta_f):
        # print('Anets')
        #####################################################
        nodes_perP1 = [[] for i in xrange(rmode.num_cores1)]
        ii = 0
        for c in cluster:
            for node in cluster[c]:
                nodes_perP1[ii].append(node)
                ii = (ii+1) % rmode.num_cores1

        nodes_perP2 = [[] for i in xrange(rmode.num_cores2)]
        ii = 0
        for c in cluster:
            for node in cluster[c]:
                nodes_perP2[ii].append(node)
                ii = (ii+1) % rmode.num_cores2

        num_f = len(self.features[self.features.keys()[0]])
        #####################################################
        # Create k clusters using CN
        current_delta_lambda = 0
        for items in link_score:
            if self.labels[items[0]] == self.labels[items[1]]:
                current_delta_lambda += items[2]
        # current_score = get_score(current_delta_lambda, self.estimate_lambda0, current_delta_f, self.f0)
        current_delta_lambda /= self.estimate_lambda0
        best_score, best_delta_lambda, best_delta_f, best_assignment, best_centroid, best_cluster = \
            self.get_best_parameters(current_score, current_delta_lambda, current_delta_f, [], [], cluster)
        ss = time.time()
        current_score, current_delta_lambda, current_delta_f, current_centroid, \
        current_cluster, current_cluster_delta_f, current_total_delta_f, current_size_delta_f, \
        current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster = \
            self.get_current_parameters(best_score,best_delta_lambda, best_delta_f, best_assignment, best_centroid,
                                        best_cluster, [], [], [], 0, 0, {}, {}, nodes_perP1, {}, [], {})
        cluster_Count = len(current_cluster)

        # Loop through the nodes until a local optimum
        loop_Counter = 0
        total_start_time = time.time()
        start_time = 0
        end_time = 0

        while loop_Counter < rmode.max_itr:
            start_time = time.time()
            print start_time
            # print('while True#####################################')
            print 'itr_Counter: ' + str(loop_Counter)
            # Start counting loops
            loop_Counter += 1
            # copy the current assignment to temporal parameters
            # For every point in the dataset ...
            # index_perm = np.random.permutation(range(1, len(self.nodes) + 1))
            print 'main part'
            if not rmode.parallel2:
                global_score, global_node_to_supper_node, global_centroid_list, global_cluster, global_index, \
                global_moved_nodes = inside_itr([self, self.nodes, current_centroid, current_cluster, cluster_Count,
                                                 link_score, link_score_dict, current_cluster_delta_f, current_total_delta_f,
                                                 current_size_delta_f, best_score, current_assignment,
                                                 current_node2C_dist, pure_delta_f, 0,
                                                 current_delta_lambda*self.estimate_lambda0, next_best_cluster])
            else:
                inp = []
                for cpu_indx in range(len(nodes_perP2)):
                    index_perm = nodes_perP2[cpu_indx]
                    inp.append([self, index_perm, current_centroid, current_cluster, cluster_Count,link_score, link_score_dict,
                                current_cluster_delta_f, current_total_delta_f, current_size_delta_f, best_score,
                                current_assignment, current_node2C_dist, pure_delta_f, cpu_indx,
                                current_delta_lambda*self.estimate_lambda0, next_best_cluster])

                try:
                    out = pool.map(inside_itr, inp)
                except KeyboardInterrupt:
                    'terminate all...'
                    pool.terminate()

                v1, v2, v3, v4, v5, v6 = zip(*out)
                global_score = {}
                global_node_to_supper_node = {}
                global_centroid_list = {}
                global_cluster = {}
                global_index = {}
                global_moved_nodes = {}
                Len = len(nodes_perP2)
                for i in range(Len):
                    for key in v1[i].keys():
                        global_score[key] = v1[i][key]
                        global_node_to_supper_node[key] = v2[i][key]
                        global_centroid_list[key] = v3[i][key]
                        global_cluster[key] = v4[i][key]
                        global_index[key] = v5[i][key]
                        global_moved_nodes[key] = v6[i][key]

            for node in global_score.keys():
                [g_score, delta_lambda, delta_f] = global_score[node]
                node_to_supper_node = global_node_to_supper_node[node]
                centroid_list = global_centroid_list[node]
                cluster = global_cluster[node]

                if g_score < best_score:
                    best_score, best_delta_lambda, best_delta_f, best_assignment, best_centroid, best_cluster = \
                        self.get_best_parameters(g_score, delta_lambda, delta_f,
                                                 node_to_supper_node, centroid_list, cluster)
                    index1, index2 = global_index[node]
                    moved_nodes = global_moved_nodes[node]

            if best_score < current_score:
                print 'updating'
                current_score, current_delta_lambda, current_delta_f, current_centroid, \
                current_cluster, current_cluster_delta_f, current_total_delta_f, current_size_delta_f, \
                current_assignment, current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster \
                    = self.get_current_parameters(best_score, best_delta_lambda, best_delta_f, best_assignment,
                                                  best_centroid, best_cluster, current_cluster_delta_f,
                                                  current_total_delta_f, current_size_delta_f, index1, index2,
                                                  current_node2C_dist, old_lambda_dict, nodes_perP2, pure_delta_f,
                                                  moved_nodes, next_best_cluster)

            else:
                print "Converged after %s iterations" % loop_Counter
                end_time = time.time()
                print end_time
                print 'itr:\t' + str(loop_Counter - 1) + '\ttime = ' + str(end_time - start_time)
                break
            end_time = time.time()
            print end_time
            print 'itr:\t' + str(loop_Counter - 1) + '\ttime = ' + str(end_time - start_time)

        total_end_time = time.time()
        print 'total itr time = ' + str(total_end_time - total_start_time)
        print 'total Anet time = ' + str(total_end_time - ss)
        with open(self.att_time_per_itr, 'w') as f:
            f.write('itr:\t' + str(loop_Counter - 1) + '\t time = ' + str(end_time - start_time))

        with open(self.att_time_all_itr, 'w') as f:
            f.write('total itr time = ' + str(total_end_time - total_start_time))
        self.labels = current_assignment
        print end_time
        return current_cluster

    def clean_up_empty(self, old_super_node, new_super_node, node, centroid_list, node_to_supper_node,
                       current_node2C_dist, cluster, moved_nodes):
        cluster[new_super_node + 1].append(node)
        node_to_supper_node[node] = new_super_node
        worst_d = rmode.bignum
        worst_node = node
        for n in cluster[new_super_node + 1]:
            # d, node_deltaf, node_delta_lambda = get_Distance2cluster(n, self, super_node_list, old_super_node,
            #                                                          new_super_node, cluster_delta_f)
            # print ('old = ') + str([d, node_deltaf, node_delta_lambda])
            try:
                d, node_deltaf, node_delta_lambda = current_node2C_dist[n][old_super_node]
            except KeyError:
                d = rmode.bignum
                node_delta_lambda = rmode.bignum
                node_deltaf = rmode.bignum
            # print ('old = ') + str([d, node_deltaf, node_delta_lambda])
            # d = get_distance(self.features[n], super_node_list[new_super_node].centroid)
            if d < worst_d:
                worst_node = n
                worst_d = d

        cluster[new_super_node + 1].remove(worst_node)
        moved_nodes.append(worst_node)
        cluster, node_to_supper_node, moved_nodes = self.clean_up(new_super_node, old_super_node, worst_node,
                                                                  centroid_list, node_to_supper_node, cluster,
                                                                  moved_nodes)

        return cluster, node_to_supper_node, moved_nodes

    def clean_up(self, old_super_node, new_super_node, node, centroid_list, node_to_supper_node, cluster,
                 moved_nodes):
        """
        :return: This function find a valid graph which is close to the input graph
        """
        cluster[new_super_node + 1].append(node)
        node_to_supper_node[node] = new_super_node
        Subgraph = self.graph.subgraph(cluster[old_super_node + 1])
        if not nx.is_connected(Subgraph):
            # print('clean_up')
            cc_list = list(nx.connected_component_subgraphs(Subgraph))
            cc_nodes = {}
            for ii in range(len(cc_list)):
                # print 'ii in range(len(cc_list))'
                cc = cc_list[ii]
                cc_nodes[ii] = {}
                for i in cc.nodes():
                    cc_nodes[ii][i] = i

            best_key, new_score = self.estimate_component_evaluation(cc_nodes, old_super_node, new_super_node,
                                                                         centroid_list, cluster)
            # remove the c[i] with highest distance.
            new_score.pop(best_key)
            # merge the rest of c[i]s
            for key, value in new_score.iteritems():
                cc = cc_nodes[key]
                for key_n, value_n in cc.iteritems():
                    cluster[old_super_node + 1].remove(value_n)
                    cluster[new_super_node + 1].append(value_n)
                    node_to_supper_node[value_n] = new_super_node
                    moved_nodes.append(value_n)
        return cluster, node_to_supper_node, moved_nodes

    def estimate_component_evaluation(self, cc_nodes, old_super_node, new_super_node, centroid_list, cluster):

        new_score = {}
        best_key = 0
        best_score = -99999
        for key, value in cc_nodes.iteritems():
            node_old = cluster[old_super_node + 1][:]
            tmp_deltaf_old = 0
            tmp_deltaf_new = 0
            tmp_deltal_old = 0
            tmp_deltal_new = 0
            for key1, value1 in value.iteritems():
                node_old.remove(value1)

            for key1, value1 in value.iteritems():
                tmp_deltaf_old += get_distance(self.features[value1], centroid_list[old_super_node + 1])
                tmp_deltaf_new += get_distance(self.features[value1], centroid_list[new_super_node + 1])
                for n in node_old:
                    try:
                        tmp_deltal_old += self.delta_lambda[str([value1, n])]
                    except KeyError:
                        try:
                            tmp_deltal_old += self.delta_lambda[str([n, value1])]
                        except KeyError:
                            continue

                for n in cluster[new_super_node + 1]:
                    try:
                        tmp_deltal_new += self.delta_lambda[str([value1, n])]
                    except KeyError:
                        try:
                            tmp_deltal_new += self.delta_lambda[str([n, value1])]
                        except KeyError:
                            continue
            Delta_Delta_f = -tmp_deltaf_old + tmp_deltaf_new
            Delta_Delta_lambda = -tmp_deltal_old + tmp_deltal_new
            new_score[key] = get_score(Delta_Delta_lambda, self.lambda0, Delta_Delta_f, self.f0)

            if new_score[key] > best_score:
                best_score = new_score[key]
                best_key = key
        return best_key, new_score

    def estimate_component_evaluation_balance(self, cc_nodes, old_super_node, new_super_node, centroid_list,
                                      pure_delta_f, cluster):
        Deltaf_old = 0
        for n in cluster[old_super_node + 1]:
            try:
                Deltaf_old += pure_delta_f[n][old_super_node + 1]
            except KeyError:
                Deltaf_old += get_distance(self.features[n], centroid_list[old_super_node + 1])

        Deltaf_new = 0
        for n in cluster[new_super_node + 1]:
            try:
                Deltaf_new += pure_delta_f[n][new_super_node + 1]
            except KeyError:
                Deltaf_new += get_distance(self.features[n], centroid_list[new_super_node + 1])

        size_old = len(cluster[old_super_node + 1])
        size_new = len(cluster[new_super_node + 1])
        new_score = {}
        best_key = 0
        best_score = -99999
        for key, value in cc_nodes.iteritems():
            node_old = cluster[old_super_node + 1][:]
            tmp_deltaf_old = 0
            tmp_deltaf_new = 0
            tmp_deltal_old = 0
            tmp_deltal_new = 0
            g_size = len(value)
            for key1, value1 in value.iteritems():
                node_old.remove(value1)

            for key1, value1 in value.iteritems():
                tmp_deltaf_old += get_distance(self.features[value1], centroid_list[old_super_node + 1])
                tmp_deltaf_new += get_distance(self.features[value1], centroid_list[new_super_node + 1])
                for n in node_old:
                    try:
                        tmp_deltal_old += self.delta_lambda[str([value1, n])]
                    except KeyError:
                        try:
                            tmp_deltal_old += self.delta_lambda[str([n, value1])]
                        except KeyError:
                            continue

                for n in cluster[new_super_node + 1]:
                    try:
                        tmp_deltal_new += self.delta_lambda[str([value1, n])]
                    except KeyError:
                        try:
                            tmp_deltal_new += self.delta_lambda[str([n, value1])]
                        except KeyError:
                            continue
            Delta_Delta_f = -tmp_deltaf_old - Deltaf_old + tmp_deltaf_new + Deltaf_new
            Delta_Delta_lambda = -tmp_deltal_old + tmp_deltal_new
            new_score[key] = get_score(Delta_Delta_lambda, self.lambda0, Delta_Delta_f, self.f0)

            if new_score[key] > best_score:
                best_score = new_score[key]
                best_key = key
        return best_key, new_score

    def real_component_evaluation(self, cc_nodes, tmp_label, tmp_cluster, old_super_node, new_super_node, link_score):
        new_score = {}
        best_key = 0
        best_score = 99999
        for key, value in cc_nodes.iteritems():
            tmp_tmp_label = copy.deepcopy(tmp_label)
            tmp_tmp_cluster = copy.deepcopy(tmp_cluster)
            for key1, value1 in value.iteritems():
                tmp_tmp_label[value1] = old_super_node
                tmp_tmp_cluster[new_super_node + 1].remove(value1)
                tmp_tmp_cluster[old_super_node + 1].append(value1)
            new_score[key], x, x = self.score(tmp_tmp_cluster, link_score, tmp_tmp_label)
            if new_score[key] < best_score:
                best_score = new_score[key]
                best_key = key
        return best_key, new_score

    def get_eig_file(self, G_dir): #, d_c, map_c
        with open(G_dir) as f:
            lines = f.readlines()
            lines.pop(0)
            g = {}
            n_map = {}
            n_num = 1
            for line in lines:
                items = line.split('\t')
                try:
                    source = int(items[0])
                    target = int(items[1])
                    try:
                        tmp = n_map[source]
                    except KeyError:
                        n_map[source] = n_num
                        n_num += 1
                    try:
                        tmp = n_map[target]
                    except KeyError:
                        n_map[target] = n_num
                        n_num += 1
                    weight = float(items[2])
                    try:
                        tmp = g[source][target]
                        tmp.append(weight)
                        g[source][target] = tmp
                    except KeyError:
                        try:
                            tmp = g[target][source]
                            tmp.append(weight)
                            g[target][source] = tmp
                        except KeyError:
                            try:
                                tmp = g[source]
                                g[source][target] = [weight]
                            except KeyError:
                                g[source] = {}
                                g[source][target] = [weight]
                except ValueError:
                    continue
        _graph = []
        # with open(d_c, 'w') as f:
        for key1 in g:
            for key2, value in g[key1].iteritems():
                # f.write(str(n_map[key1]) + '\t' + str(n_map[key2]) + '\t')
                if len(value) >= 2:
                    # f.write(str(value[0]) + '\t' + str(value[1]) + '\n')
                    _graph.append([n_map[key1], n_map[key2], value[0], value[1]])
                else:
                    # f.write(str(value[0]) + '\t0' + '\n')
                    _graph.append([n_map[key1], n_map[key2], value[0], 0])
        # with open(map_c, 'w') as f:
        #     for key, value in n_map.iteritems():
        #         f.write(str(key) + ' : ' + str(value) + '\n')

        return _graph

    def calculate_eig(self, link_score, labels):
        score = []
        for items in link_score:
            if labels[items[0]] == labels[items[1]]:
                score.append([items[0], items[1], items[2]])

        with open(self.tmp_score_dir, 'w') as f:
            for line in score:
                f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')
        # merge
        subprocess.call(
            ["src/CoarseNet", self.coarse_input_dir, self.tmp_score_dir, str(rmode.percent), '0',
             self.tmp_output_dir,
             self.tmp_map_dir, self.tmp_time_dir])
        # calucate eig
        _graph = self.get_eig_file(self.tmp_output_dir) #, d_c, map_c
        eig_vec, vec2, eig, G, tmp1 = gES.getEigenScore(_graph, self.tol)
        eig = eig[0]
        return eig, eig_vec

    def moved2dict(self, moved_nodes):
        moved_nodes_dict = {}
        for key in moved_nodes:
            moved_nodes_dict[key] = {}
            for item in moved_nodes[key]:
                moved_nodes_dict[key][item] = True
        for key in moved_nodes_dict['2new']:
            try:
                tmp = moved_nodes_dict['2old'][key]
            except KeyError:
                tmp = False
            if tmp:
                del moved_nodes_dict['2new'][key]
                del moved_nodes_dict['2old'][key]
                break
        return moved_nodes_dict

    def calculate_eig_estimate(self, link_score, link_score_dict, labels, moved_nodes, current_delta_lambda,
                               old_indx, new_indx):
        eig = current_delta_lambda

        moved_nodes_dict = self.moved2dict(moved_nodes)

        for key in moved_nodes_dict:
            Is = {}
            for item in moved_nodes_dict[key]:
                neighbours = nx.neighbors(self.graph, item)
                for n in neighbours:
                    try:
                        Is['2old'] = moved_nodes_dict['2old'][n]
                    except KeyError:
                        Is['2old'] = False
                    try:
                        Is['2new'] = moved_nodes_dict['2new'][n]
                    except KeyError:
                        Is['2new'] = False

                    if (labels[n] == old_indx or labels[n] == new_indx) and (not (Is[key])):
                        try:
                            e = [n, item]
                            tmp = link_score_dict[str(e)]
                        except KeyError:
                            try:
                                tmp = link_score_dict[str([e[1], e[0]])]
                            except KeyError:
                                tmp = 0
                        if key == '2new' and labels[n] == old_indx and (not Is['2old']):
                            eig -= tmp
                        if key == '2new' and labels[n] == new_indx:
                            eig += tmp
                        if key == '2old' and labels[n] == old_indx:
                            eig += tmp
                        if key == '2old' and labels[n] == new_indx and (not Is['2new']):
                            eig -= tmp
        # eig1 = eig
        # eig = 0
        # for items in link_score:
        #     if labels[items[0]] == labels[items[1]]:
        #         eig += items[2]
        # if abs(eig1 - eig) > 1e-6:
        #     print 'hi'
        return eig

    def calculate_f0(self):
        feature_arr = []
        for key, value in self.features.iteritems():
            feature_arr.append(value)
        feature_arr = np.array(feature_arr)
        center = feature_arr.mean(0)
        delta_f = 0

        for key, value in self.features.iteritems():
            delta_f += get_distance(value, center)
        self.f0 = delta_f
        self.feature_arr = feature_arr

    def update_supernode_delta(self, super_node_list, index, center):
        super_node = super_node_list[index]
        total_delta = 0
        for i in range(len(super_node)):
            node1 = super_node[i]
            total_delta += get_distance(self.features[node1], center)

        return total_delta

    def calculate_supernode_delta1(self, super_node_list, index):
        try:
            super_node = super_node_list[index]
            sub_edges = self.graph.subgraph(super_node).edges()
            total_delta = 0
            for node1, node2 in sub_edges:
                u = np.linalg.norm(self.features[node1])
                v = np.linalg.norm(self.features[node2])
                if u == 0 or v == 0:
                    total_delta += 0
                else:
                    total_delta += (1 - get_distance(self.features[node1], self.features[node2]))
                    # total_delta += (1 - spatial.distance.cosine(self.features[node1], self.features[node2]))
        except KeyError:
            total_delta = 0
            sub_edges = []
        return total_delta, len(sub_edges)

    def calculate_supernode_delta(self, super_node_list, index):
        super_node = super_node_list[index]
        features_arr = []
        for node in super_node:
            try:
                features_arr.append(self.features[node])
            except KeyError:
                print node
        features_arr = np.array(features_arr)
        center = features_arr.mean(0)

        # compute \delta f
        total_delta = 0
        for i in range(len(super_node)):
            node1 = super_node[i]
            total_delta += get_distance(self.features[node1], center)

        return total_delta

    def estimate_score(self, total_delta_f, size_delta_f, node_cluster, link_score, link_score_dict, current_assignment,
                       old_cluster, super_node_Index, cluster_delta_f, moved_nodes, current_delta_lambda,
                       current_centroid, pure_delta_f):
        delta_f = total_delta_f
        # Compute Delta f
        center = {}
        center[old_cluster + 1] = current_centroid[old_cluster + 1]
        center[super_node_Index + 1] = current_centroid[super_node_Index + 1]
        if False:
            center = self.update_centroid(center, node_cluster, [old_cluster + 1, super_node_Index + 1], moved_nodes)
            for index in [old_cluster, super_node_Index]:
                delta_f -= cluster_delta_f[index]
                delta_f += self.update_supernode_delta(node_cluster, index + 1, center[index + 1])
        else:
            center_old = center[old_cluster + 1]
            center_new = center[super_node_Index + 1]

            moved_nodes_dict = self.moved2dict(moved_nodes)
            # compute delta f
            delta = self.get_estimate_delta_f('2new', moved_nodes_dict, pure_delta_f, old_cluster, center_new)
            delta_f += delta
            delta = self.get_estimate_delta_f('2old', moved_nodes_dict, pure_delta_f, super_node_Index, center_old)
            delta_f += delta

        delta_lambda = self.calculate_eig_estimate(link_score, link_score_dict, current_assignment, moved_nodes,
                                                   current_delta_lambda, old_cluster, super_node_Index)

        # Compute Score
        score = get_score(delta_lambda, self.estimate_lambda0, delta_f, self.f0)
        return score, delta_lambda / self.estimate_lambda0, normalize(delta_f, self.f0)

    def get_estimate_delta_f_balanced(self, key, moved_nodes_dict, pure_delta_f, index, new_indx, size_old, size_new, center, cluster_delta_f):
        f_old = 0
        f_new = 0
        for a in moved_nodes_dict[key]:
            f_old += pure_delta_f[a][index + 1]
            f_new += get_distance(self.features[a], center)

        f_old = cluster_delta_f[index] + f_old
        f_new = cluster_delta_f[new_indx] + f_new
        delta = f_new - f_old
        return delta

    def get_estimate_delta_f(self, key, moved_nodes_dict, pure_delta_f, index, center):
        f_old = 0
        f_new = 0
        for a in moved_nodes_dict[key]:
            f_old += pure_delta_f[a][index + 1]
            f_new += get_distance(self.features[a], center)
        delta = f_new - f_old
        return delta

    def score(self, node_cluster, link_score, current_assignment):
        delta_f = 0

        # Compute Delta f
        for index in range(len(node_cluster)):
            delta_f += self.calculate_supernode_delta(node_cluster, index + 1)

        # Compute Delta eig
        eig, eig_vec = self.calculate_eig(link_score, current_assignment)
        delta_lambda = abs(eig - self.lambda0)

        # Compute Score
        score = get_score(delta_lambda, self.lambda0, delta_f, self.f0)
        return score, delta_lambda / self.lambda0, normalize(delta_f, self.f0)

    def estimate_delta_lambda(self, node_to_supper_node, link_score):
        delta_lambda = 0
        for node1, node2, v in link_score:
            if node_to_supper_node[node1] == node_to_supper_node[node2]:
                delta_lambda += v
        return delta_lambda

    def save_score(self, score_CN, delta_lambda_CN, delta_f_CN, mode, algorithm):
        with open(self.score_dir, mode) as f:
            if mode == 'w':
                f.write('Alg\tDelta_lambda\tDelta_f\tScore\n')
            f.write(algorithm + '\t' + str(delta_lambda_CN) + '\t' + str(delta_f_CN) + '\t' + str(score_CN) + '\n')

    def merge(self):
        subprocess.call(
            ["src/CoarseNet", self.coarse_input_dir, self.att_score_dir, str(rmode.percent), '0',
             self.coarse_att_output_dir,
             self.att_map_dir, self.att_time_dir])
        with open(self.att_labels_dir, 'w') as f:
            f.write('node\tlabel\n')
            for key, value in self.labels.iteritems():
                f.write(str(key) + '\t' + str(value) + '\n')


def normalize(v, max_v):
    v /= max_v
    return v


def inside_itr(inp):
    G, index_perm, current_centroid, current_cluster, cluster_Count, link_score, link_score_dict, current_cluster_delta_f, \
    current_total_delta_f, current_size_delta_f, best_score, current_assignment, current_node2C_dist, pure_delta_f, \
    cpu_idx, current_delta_lambda, next_best_cluster = inp
    global_score = {}
    global_node_to_supper_node = {}
    global_centroid_list = {}
    global_cluster = {}
    global_index = {}
    global_moved_nodes = {}
    for key in index_perm:
        node = G.nodes[key]
        # Get the distance between that point and the all data points of the first cluster
        node_to_supper_node = current_assignment.copy()
        centroid_list = current_centroid
        cluster = deepcopy_cluster(current_cluster)
        old_cluster = node_to_supper_node[node]
        cluster[old_cluster + 1].remove(node)

        smallest_Delta_score, super_node_Index = next_best_cluster[node]

        recalc_flag = False
        moved_nodes = {}
        moved_nodes['2old'] = []
        moved_nodes['2new'] = [node]
        if len(cluster[old_cluster + 1]) == 0:
            recalc_flag = True
            cluster, node_to_supper_node, moved_nodes['2old'] = G.clean_up_empty(old_cluster, super_node_Index - 1,
                                                                                 node, centroid_list,
                                                                                 node_to_supper_node,
                                                                                 current_node2C_dist,
                                                                                 cluster, moved_nodes['2old'])

        elif not (old_cluster + 1) == super_node_Index:
            recalc_flag = True
            cluster, node_to_supper_node, moved_nodes['2new'] = G.clean_up(old_cluster, super_node_Index - 1, node,
                                                                           centroid_list, node_to_supper_node, cluster,
                                                                           moved_nodes['2new'])

        if recalc_flag:
            g_score, delta_lambda, delta_f = G.estimate_score(current_total_delta_f, current_size_delta_f,
                                                              cluster, link_score, link_score_dict, node_to_supper_node,
                                                              old_cluster, super_node_Index - 1,
                                                              current_cluster_delta_f, moved_nodes,
                                                              current_delta_lambda, centroid_list, pure_delta_f)

            if g_score < best_score:
                global_score[cpu_idx] = [g_score, delta_lambda, delta_f]
                global_node_to_supper_node[cpu_idx] = node_to_supper_node
                global_centroid_list[cpu_idx] = centroid_list
                global_cluster[cpu_idx] = cluster
                index1 = old_cluster
                index2 = super_node_Index - 1
                global_index[cpu_idx] = [index1, index2]
                global_moved_nodes[cpu_idx] = moved_nodes
                best_score = g_score

        else:
            cluster[old_cluster + 1].append(node)

    return global_score, global_node_to_supper_node, global_centroid_list, global_cluster, global_index, \
           global_moved_nodes


def p_ucp(inp):
    G, nodes, current_cluster, changed_node_list, current_centroid, index1, index2, current_cluster_delta_f, \
    current_node2C_dist, old_lambda_dict, current_assignment, pure_delta_f, next_best_cluster= inp
    for node in nodes:
        tmp_dict = {}
        old_cluster = current_assignment[node]
        current_cluster_tmp = copy.deepcopy(current_cluster[old_cluster + 1])
        current_cluster_tmp.remove(node)
        if node in changed_node_list:
            old_lambda_dict = get_old_lambda(G, old_lambda_dict, current_cluster[old_cluster + 1], node)
            cluster_index = range(len(current_centroid))
            pure_delta_f[node][old_cluster + 1] = get_distance(G.features[node], current_centroid[old_cluster + 1])
        else:
            cluster_index = [index1, index2]
        old_cluster = current_assignment[node]

        for c_index in cluster_index:
            if c_index == old_cluster:
                Delta_score, node_deltaf, node_delta_lambda, pure_delta_f = get_Distance2cluster_parallel(node, G, current_centroid,
                                                                                   c_index, old_cluster,
                                                                                   old_lambda_dict[node],
                                                                                   current_cluster_tmp,
                                                                                   pure_delta_f)
            else:
                Delta_score, node_deltaf, node_delta_lambda, pure_delta_f = get_Distance2cluster_parallel(node, G, current_centroid,
                                                                                            c_index, old_cluster,
                                                                                            old_lambda_dict[node],
                                                                                            current_cluster[c_index + 1],
                                                                                            pure_delta_f)
            if Delta_score < rmode.bignum:
                tmp_dict[c_index] = [Delta_score, node_deltaf, node_delta_lambda]
        current_node2C_dist[node] = tmp_dict
        try:
            changed_node_list[node]
            next_best_cluster[node] = get_next_best_cluster(node, current_assignment[node], current_cluster,
                                                            current_node2C_dist, len(current_cluster))
        except KeyError:
            if next_best_cluster[node][1] in [index1, index2]:
                next_best_cluster[node] = get_next_best_cluster(node, current_assignment[node], current_cluster,
                                                                current_node2C_dist, len(current_cluster))
            else:
                next_best_cluster[node] = update_next_best_cluster(next_best_cluster[node], node, [index1, index2],
                                                                   current_node2C_dist)
    return current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster


def update_next_best_cluster(next_best_cluster, node, cluster_index, current_node2C_dist):
    # Get the distance between that point and the all data points of the first cluster
    smallest_Delta_score, super_node_Index = next_best_cluster
    # For the remainder of the clusters ...
    for i in cluster_index:
        i -= 1
        try:
            Delta_score, node_deltaf, node_delta_lambda = current_node2C_dist[node][i + 1]
        except KeyError:
            Delta_score = rmode.bignum
            node_delta_lambda = rmode.bignum
            node_deltaf = rmode.bignum

        if Delta_score < smallest_Delta_score:
            smallest_Delta_score = Delta_score
            super_node_Index = i + 2

    return [smallest_Delta_score, super_node_Index]


def p_fcp(inp):
    G, nodes, current_cluster, current_centroid, current_cluster_delta_f, current_assignment, pure_delta_f = inp
    current_node2C_dist = {}
    old_lambda_dict = {}
    next_best_cluster = {}
    for node in nodes:
        tmp_dict = {}
        current_node2C_dist[node] = {}
        old_cluster = current_assignment[node]
        pure_delta_f[node] = {}
        pure_delta_f[node][old_cluster + 1] = get_distance(G.features[node], current_centroid[old_cluster + 1])

        current_cluster_tmp = copy.deepcopy(current_cluster[old_cluster + 1])
        current_cluster_tmp.remove(node)
        old_lambda_dict = get_old_lambda(G, old_lambda_dict, current_cluster_tmp, node)
        cluster_index = range(len(current_centroid))
        for c_index in cluster_index:
            if c_index == old_cluster:
                Delta_score, node_deltaf, node_delta_lambda, pure_delta_f = get_Distance2cluster_parallel(node, G, current_centroid,
                                                                                            c_index, old_cluster,
                                                                                            old_lambda_dict[node],
                                                                                            current_cluster_tmp,
                                                                                            pure_delta_f)
            else:
                Delta_score, node_deltaf, node_delta_lambda, pure_delta_f = get_Distance2cluster_parallel(node, G, current_centroid,
                                                                                            c_index, old_cluster,
                                                                                            old_lambda_dict[node],
                                                                                            current_cluster[c_index + 1],
                                                                                            pure_delta_f)
            if Delta_score < rmode.bignum:
                tmp_dict[c_index] = [Delta_score, node_deltaf, node_delta_lambda]
        current_node2C_dist[node] = tmp_dict
        next_best_cluster[node] = get_next_best_cluster(node, current_assignment[node], current_cluster, current_node2C_dist, len(current_cluster))
    return current_node2C_dist, old_lambda_dict, pure_delta_f, next_best_cluster


def paralle_dist_2_centroid(inp):
    assignment, features, centroid_list, nodes, Indx = inp
    f = {}
    for i in Indx:
        f[i] = 0
    for node in nodes:
        indx = assignment[node]+1
        center = centroid_list[indx]
        f[indx-1] += get_distance(features[node], center)
    return f


def get_cluster_delta(inp):
    labels, features, centroid_list, cluster, nodes = inp
    cluster_delta_f = {}
    f = {}
    for i in cluster:
        f[i - 1] = 0
    for n in nodes:
        Indx = labels[n]
        center = centroid_list[Indx]
        f[Indx - 1] += get_distance(features[n], center)
        cluster_delta_f[Indx - 1] = f[Indx - 1]

    return cluster_delta_f


def get_next_best_cluster(node, old_cluster, current_cluster, current_node2C_dist, cluster_Count):
    # Get the distance between that point and the all data points of the first cluster
    cluster = deepcopy_cluster(current_cluster)
    cluster[old_cluster + 1].remove(node)
    cluster_index = range(cluster_Count)

    try:
        smallest_Delta_score, smallest_node_deltaf, smallest_node_delta_lambda = current_node2C_dist[node][old_cluster]
    except KeyError:
        smallest_Delta_score = rmode.bignum
        smallest_node_delta_lambda = rmode.bignum
        smallest_node_deltaf = rmode.bignum
    # Set the cluster this point belongs to
    super_node_Index = old_cluster + 1
    cluster_index.pop(old_cluster)
    # For the remainder of the clusters ...
    for i in cluster_index:
        i -= 1
        try:
            Delta_score, node_deltaf, node_delta_lambda = current_node2C_dist[node][i + 1]
        except KeyError:
            Delta_score = rmode.bignum
            node_delta_lambda = rmode.bignum
            node_deltaf = rmode.bignum

        if Delta_score < smallest_Delta_score:
            smallest_Delta_score = Delta_score
            super_node_Index = i + 2
            smallest_node_deltaf = node_deltaf
            smallest_node_delta_lambda = node_delta_lambda

    return [smallest_Delta_score, super_node_Index]


def deepcopy_cluster(cluster1):
    cluster2 = {}
    for key, value in cluster1.iteritems():
        cluster2[key] = value[:]
    return cluster2


def calculateCentroid_parallel(nodes, features):
    """
    :param self:
    :param points:
    :return: center
     Finds a virtual center point for a group of n-dimensional points
    """
    if len(nodes) > 1:
        features_arr = np.empty(shape=[0,])

        ####################
        inp = []
        for s, t in rmode.features_perP:
            inp.append([nodes, features, s, t])

        try:
            out = pool.map(sup_cp, inp)
        except KeyboardInterrupt:
            'terminate all...'
            pool.terminate()
        ####################
        for ll in out:
            features_arr = np.concatenate([features_arr, ll], axis=0)

        return features_arr
    else:
        tmp = np.array(features[nodes[0]])
        return np.array(features[nodes[0]])


def sup_cp(inp):
    nodes, features, s, t = inp
    local_f_arr = []
    for node in nodes:
        # tmp1 = features[node]
        local_f_arr.append(features[node][s:t])
    local_f_arr = np.array(local_f_arr)
    ll = local_f_arr.mean(0)

    return ll


def calculateCentroid(nodes, features):
    """
    :param self:
    :param points:
    :return: center
     Finds a virtual center point for a group of n-dimensional points
    """
    features_arr = []
    for node in nodes:
        features_arr.append(features[node])
    features_arr = np.array(features_arr)
    return features_arr.mean(0)


def get_score(delta_lambda, lambda0, delta_f, f0):
    score = 0.5 * delta_lambda / lambda0 + 0.5 * normalize(delta_f, f0)
    return score


def update_f_parallel_balance(G, cluster_delta_f, centroid_list, old_index, new_index, a, cluster_old, cluster_new,
                      pure_delta_f):
    center_old = centroid_list[old_index + 1]
    center_new = centroid_list[new_index + 1]
    size_old = len(cluster_old) + 1
    size_new = len(cluster_new) + 1

    # compute delta f
    f_old = pure_delta_f[a][old_index + 1]
    f_old += (cluster_delta_f[old_index] - pure_delta_f[a][old_index + 1])

    pure_delta_f[a][new_index + 1] = get_distance(G.features[a], center_new)
    f_new = pure_delta_f[a][new_index + 1]
    f_new += cluster_delta_f[new_index]

    delta_f = f_new - f_old
    return delta_f, pure_delta_f


def update_f_parallel(G, centroid_list, old_index, new_index, a, pure_delta_f):
    center_new = centroid_list[new_index + 1]

    # compute delta f
    f_old = pure_delta_f[a][old_index + 1]

    pure_delta_f[a][new_index + 1] = get_distance(G.features[a], center_new)
    f_new = pure_delta_f[a][new_index + 1]

    delta_f = f_new - f_old
    return delta_f, pure_delta_f


def update_f(G, centroid_list, old_index, new_index, a):
    center_old = centroid_list[old_index + 1]
    center_new = centroid_list[new_index + 1]

    # compute delta f
    f_old = get_distance(G.features[a], center_old)

    f_new = get_distance(G.features[a], center_new)

    delta_f = f_new - f_old
    return delta_f


def update_f_balance(G, cluster_delta_f, centroid_list, old_index, new_index, a, cluster, cluster_new):
    cluster_old = cluster[old_index + 1]
    center_old = centroid_list[old_index + 1]
    center_new = centroid_list[new_index + 1]
    size_old = len(cluster_old) + 1
    size_new = len(cluster_new) + 1

    # compute delta f
    tmp_d = get_distance(G.features[a], center_old)
    f_old = tmp_d
    f_old += (cluster_delta_f[old_index] - tmp_d)

    f_new = get_distance(G.features[a], center_new)
    f_new += cluster_delta_f[new_index]

    delta_f = f_new - f_old
    return delta_f


def get_old_lambda(G, old_lambda_dict, cluster_old, a):
    lambda_old = 0
    for item in cluster_old:
        try:
            lambda_old += G.delta_lambda[str([a, item])]
        except KeyError:
            try:
                lambda_old += G.delta_lambda[str([item, a])]
            except KeyError:
                continue
    old_lambda_dict[a] = lambda_old
    return old_lambda_dict


def get_Distance2cluster_parallel(a, G, centroid_list, new_index, old_index, old_lambda_dict_tmp, cluster_new, pure_delta_f):
    node_deltaf = 0
    node_deltalambda = 0
    # compute delta lambda
    lambda_new = 0
    connected = False
    for item in cluster_new:
        try:
            lambda_new += G.delta_lambda[str([a, item])]
            connected = True
        except KeyError:
            try:
                lambda_new += G.delta_lambda[str([item, a])]
                connected = True
            except KeyError:
                continue
    if old_index == new_index or len(cluster_new) == 0:
        connected = True

    delta_lambda = lambda_new - old_lambda_dict_tmp
    # compute delta score
    if connected:
        delta_f, pure_delta_f = update_f_parallel(G, centroid_list, old_index, new_index, a, pure_delta_f)
        delta_score = get_score(delta_lambda, G.lambda0, delta_f, G.f0)
        node_deltaf = normalize(delta_f, G.f0)
        node_deltalambda = delta_lambda / G.lambda0
    else:
        delta_score = rmode.bignum
        try:
            tmp = pure_delta_f[a][new_index + 1]
            del pure_delta_f[a][new_index + 1]
        except KeyError:
            new_index
    return delta_score, node_deltaf, node_deltalambda, pure_delta_f


def get_Distance2cluster(a, G, centroid_list, new_index, old_index, old_lambda_dict, cluster):
    cluster_new = cluster[new_index + 1]
    node_deltaf = 0
    node_deltalambda = 0
    # compute delta lambda
    lambda_new = 0
    connected = False
    for item in cluster_new:
        try:
            lambda_new += G.delta_lambda[str([a, item])]
            connected = True
        except KeyError:
            try:
                lambda_new += G.delta_lambda[str([item, a])]
                connected = True
            except KeyError:
                continue
    if old_index == new_index or len(cluster_new) == 0:
        connected = True

    delta_lambda = lambda_new - old_lambda_dict[a]
    # compute delta score
    if connected:
        delta_f = update_f(G, centroid_list, old_index, new_index, a)
        delta_score = get_score(delta_lambda, G.lambda0, delta_f, G.f0)
        node_deltaf = normalize(delta_f, G.f0)
        node_deltalambda = delta_lambda / G.lambda0
    else:
        delta_score = rmode.bignum
    return delta_score, node_deltaf, node_deltalambda


def get_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    norm = math.sqrt(len(b))
    D = np.linalg.norm(a - b) / norm
    return D**2


def save_cluster(dir_, cluster):
    with open(dir_, 'w') as f:
        for key, value in cluster.iteritems():
            f.write(str(key) + ':')
            for v in value:
                f.write('\t' + str(v))
            f.write('\n')


def make_edge_order(link_score, labels, att_scor_dir):
    # print('make_edge_order')
    score = []
    att_score = []
    for items in link_score:
        score.append([int(items[0]), int(items[1]), float(items[2])])
        if labels[int(items[0])] == labels[int(items[1])]:
            att_score.append([int(items[0]), int(items[1]), float(items[2])])

    with open(att_scor_dir, 'w') as f:
        for line in att_score:
            f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')


def save_result(out_dir, G, link_score, att_map):
    # print('save_final_results')
    make_edge_order(link_score, G.labels, G.att_score_dir)
    G.merge()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    label_dictionary = G.labels.copy()
    # loading the edges of att_summarization
    with open(G.coarse_att_output_dir) as f:
        att_edge = []
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            try:
                att_edge.append([int(items[0]), int(items[1]), float(items[2])])
            except ValueError:
                continue
    # update the links based on new node IDs
    for i in range(len(att_edge)):
        att_edge[i][0] = label_dictionary[att_edge[i][0]]
        att_edge[i][1] = label_dictionary[att_edge[i][1]]

    # update the feature file based on new node IDs
    feature = {}
    for key, value in att_map.iteritems():
        tmp = []
        for v in value:
            tmp.append(G.features[v])
        tmp = np.array(tmp)
        feature[key - 1] = tmp.mean(0)

    # save the new feature file
    num_dim = len(feature[0])
    with open(out_dir + '/features.txt', 'w') as f:
        f.write('Id\t')
        for i in range(num_dim):
            f.write('feature' + str(i) + '\t')
        f.write('\n')
        for key, value in feature.iteritems():
            f.write(str(key) + '\t')
            for v in value:
                f.write(str(v) + '\t')
            f.write('\n')

    with open(out_dir + '/links.txt', 'w') as f:
        f.write('source\ttarget\tweight\n')
        for n1, n2, w in att_edge:
            f.write(str(n1) + '\t' + str(n2) + '\t' + str(w) + '\n')

    save_cluster(out_dir + '/node_map.txt', att_map)


def weight(f1, f2):
    d = len(f1)
    _sum = 0
    for i in xrange(d):
        _sum += (f1[i] - f2[i]) ** 2
    if d > 0:
        return (_sum/d)
    else:
        return 0


def read_graph(links_dir, features_dir):
    G = nx.Graph()
    coarse_input = []
    with open(links_dir) as f:
        lines = f.readlines()
    for line in lines:
        items = line.split('\t')
        G.add_edge(int(items[0]), int(items[1]))
        coarse_input.append([int(items[0]), int(items[1]), float(items[2]), float(items[3])])

    features = {}
    with open(features_dir) as f:
        lines = f.readlines()
    lines.pop(0)
    for line in lines:
        line = line.replace('\t\n', '')
        items = line.split('\t')
        node = int(items.pop(0))
        f_vec = [float(i) for i in items]
        features[node] = f_vec
    return G, features, coarse_input


def make_initialization_file(G, lambdas, intermediate_results_dir, fs0, features):
    ws = []
    for n1, n2 in G.edges():
        f = weight(features[n1], features[n2])
        f /= fs0
        try:
            l = lambdas[str([n1, n2])]
        except KeyError:
            try:
                l = lambdas[str([n2, n1])]
            except KeyError:
                print str([n1, n2])
                l = 0
        w = 0.5 * l + 0.5 * f
        ws.append([n1, n2, w])
    ws = np.array(ws)
    ws = ws[ws[:, 2].argsort()]

    CN_file = open(intermediate_results_dir + 'fl_score.txt', 'w')
    for n1, n2, w in ws:
        CN_file.write(str(int(n1)) + '\t' + str(int(n2)) + '\t' + str(w) + '\n')
    CN_file.close()


def make_feature_file(G, input_dir, features):
    fs = []
    fs0 = 0
    for n1, n2 in G.edges():
        w = weight(features[n1], features[n2])
        fs.append([n1, n2, w])
        fs0 += w
    fs = np.array(fs)
    fs = fs[fs[:, 2].argsort()]

    CN_file = open(input_dir, 'w')
    for n1, n2, w in fs:
        CN_file.write(str(int(n1)) + '\t' + str(int(n2)) + '\t' + str(w) + '\n')
    CN_file.close()

    return fs0


def get_lambda(input_dir, coarse_input):
    lambdas = {}
    link_score_dir = input_dir + 'score.txt'
    lambda0_dir = input_dir + 'lambda0.txt'
    try:
        with open(lambda0_dir) as f:
            real_lambda0 = float(f.readline())
        with open(link_score_dir) as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            items = line.split('\t')
            n1 = int(items[0])
            n2 = int(items[1])
            v = float(items[2])
            lambdas[str([n1, n2])] = v/real_lambda0
    except IOError:
        link_score, lambda0 = gES.main(coarse_input, link_score_dir, lambda0_dir, 1e-4)
        real_lambda0 = lambda0[0]

        for n1, n2, score in link_score:
            lambdas[str([n1, n2])] = score/real_lambda0
    return lambdas


def inirialization(links_dir, features_dir, intermediate_results_dir, p):
    G, features, coarse_input = read_graph(links_dir, features_dir)
    lambdas = get_lambda(intermediate_results_dir, coarse_input)
    fs0 = make_feature_file(G, intermediate_results_dir + 'f_scores.txt', features)
    make_initialization_file(G, lambdas, intermediate_results_dir, fs0, features)

    coarse_input_dir = links_dir

    subprocess.call(
        ["src/CoarseNet", coarse_input_dir, intermediate_results_dir + 'fl_score.txt', str(p), '0',
         intermediate_results_dir + 'coarse.txt', intermediate_results_dir + 'final_map.txt',
         intermediate_results_dir + 'acn_time.txt'])


def main(r):
    global rmode
    global pool
    rmode = r

    # Get the directories
    _dir = directory()
    _dir.get_dir(rmode)

    # Load data
    data = Data(_dir)
    data.load_data(_dir)

    # Make an att_graph object
    G = Att_graph(data.nodes, data.edges, data.nodes.copy(), data.features, _dir, rmode.percent, rmode.tol)
    inirialization(_dir.edge_dir, _dir.feature_dir, _dir.intermediate_results, str(rmode.percent))
    cluster, link_score, link_score_dict = G.acn()

    # Get the score of CN
    score_CN, delta_lambda_CN, delta_f_CN = G.score(cluster, link_score, G.labels)
    print '######################################################'
    print 'delta_lambda = ' + str(delta_lambda_CN)
    print 'delta_f = ' + str(delta_f_CN)

    # Run ANets
    if rmode.parallel1 or rmode.parallel2:
        num_cores = max(num_cores1, num_cores2)
        pool = multiprocessing.Pool(num_cores)
    Anets_cluster = G.anets(cluster, link_score, link_score_dict, score_CN, delta_f_CN)
    if rmode.parallel1 or rmode.parallel2:
        pool.close()
        pool.join()
    # Get the score of ANets
    score_ANets, delta_lambda_ANets, delta_f_ANets = G.score(Anets_cluster, link_score, G.labels)

    # Save the results
    G.save_score(score_ANets, delta_lambda_ANets, delta_f_ANets, 'w', 'ANets')
    save_result(_dir.output_dir, G, link_score, Anets_cluster)

    print 'Sequential results'
    print 'delta_lambda = ' + str(delta_lambda_ANets)
    print 'delta_f = ' + str(delta_f_ANets)
    print 'score = ' + str(score_ANets)


input_data = sys.argv[1]
percent = float(sys.argv[2])
num_cores1 = int(sys.argv[3])
num_cores2 = int(sys.argv[4])

max_itr = 1000  # define the maximum number of iteration before convergence
tol = float('1e-3')

rmode = Running_mode(input_data, percent, max_itr, tol, num_cores1, num_cores2)
main(rmode)
