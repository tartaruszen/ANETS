__author__ = 'Sorour E.amiri'
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spl


def getGraph(_graph):
    n = max(max(_graph)) - 1
    G = [[row[0] - 1, row[1] - 1, row[2], row[3]] for row in _graph]
    _graph = G[:]
    G = np.array(G)
    G2 = G[:, [1, 0, 3, 2]]
    G = np.concatenate((G, G2))
    i = np.array([n, n, 0, 0])
    i = i.reshape(1, 4)
    G = np.concatenate((G, i))
    # n = max(max(_graph))
    # print('G.shape: ' + str(G.shape))
    # print('n: ' + str(n))
    # print('max(max(G)): ' + str(G[:, 0:2].max()))
    n = G[:, 0:2].max()
    G_sparse = sp.coo_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n + 1, n + 1))
    return G_sparse, np.array(_graph)


def getEigenScore(_graph, tol):
    # assuming we have a list of edges E = [i j weight_i_to_j weight_j_to_i]
    # Every undirected edge appears twice
    G, G_array = getGraph(_graph)
    # eigen value
    val, vec1 = spl.eigs(G, k=1, which='LM', tol=tol)
    vec1 = np.absolute(vec1)
    val, vec2 = spl.eigs(G.transpose(), k=1, which='LM', tol=tol)
    vec2 = np.absolute(vec2)
    val = np.absolute(val)
    return vec1, vec2, val, G, G_array


def ComputeLinkScore(vec1, vec2, val, G_array):
    uv = np.multiply(vec1, vec2)
    a = G_array[:, 0]
    a = a.astype(int)
    # a = np.matrix(a)
    b = G_array[:, 1]
    b = b.astype(int)
    # b = np.matrix(b)
    beta1 = G_array[:, 2].reshape(len(b), 1)
    beta2 = G_array[:, 3].reshape(len(b), 1)
    v1 = -val * (uv[a] + uv[b])
    v2 = np.multiply(((1 + beta2) / 2), (val * vec1[a] - np.multiply(beta1, vec1[b])))
    v3 = np.multiply(((1 + beta1) / 2), (val * vec1[b] - np.multiply(beta2, vec1[a])))
    v4 = np.multiply(vec2[a], (v2 + v3))
    v5 = np.multiply(np.multiply(vec1[a], vec2[b]), beta2) + np.multiply(vec1[b], np.multiply(vec2[a], beta1))
    v6 = np.vdot(vec2, vec1) - (uv[a] + uv[b])
    delLam_vec = np.divide(-(v1 + v4 + v5), v6)
    a = a.reshape(len(b), 1)
    b = b.reshape(len(b), 1)
    delLam_sol = np.concatenate((a, b, delLam_vec), axis=1)
    delLam_sol = delLam_sol[delLam_sol[:, 2].argsort()]
    list_delLam_sol = delLam_sol
    return list_delLam_sol


def SaveResults(list_delLam_sol, val, out_file_Name, lambda0_dir):
    out_file = []
    with open(out_file_Name, 'w') as f:
        for line in list_delLam_sol:
            f.write(str(int(line[0] + 1)) + '\t' + str(int(line[1] + 1)) + '\t' + str(line[2]) + '\n')
            out_file.append([int(line[0] + 1), int(line[1] + 1), line[2]])

    with open(lambda0_dir, 'w') as f:
        f.write(str(val[0]))
    return out_file


def main(_graph, out_file_Name, lambda0_dir, tol):
    print('computing scores...')
    vec1, vec2, val, G, G_array = getEigenScore(_graph, tol)
    list_delLam_sol = ComputeLinkScore(vec1, vec2, val, G_array)
    out_file = SaveResults(list_delLam_sol, val, out_file_Name, lambda0_dir)
    return out_file, val


if __name__ == '__main__':
    graph = sys.argv[1]
    out_file_Name = sys.argv[2]
    lambda0_dir = sys.srgv[3]
    tol = sys.srgv[4]
    # graph = [[1, 2, 0.2, 0.2], [2, 3, 0.2, 0.2]]
    # infected_id = [1, 2]
    # out_file_Name = 'eig.txt'
    main(graph, out_file_Name, lambda0_dir, tol)
