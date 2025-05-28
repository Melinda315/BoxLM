import torch
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.sparse as sp

def _bi_norm_lap(adj):
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()


class DataUtils():
    def __init__(self, args):

        super(DataUtils, self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu_id}")

    def build_graph(self, train_data, visit2icd, ccs2icd, data_stat):
        print("building graph")
        n_visits = data_stat["n_visits"]
        n_ccss = data_stat["n_ccss"]
        n_nodes = data_stat["n_nodes"]
        
        train_data = train_data.copy()
        train_data[:,1] = train_data[:,1] + n_visits

        row = train_data[:,0]
        col = train_data[:,1]

        visit2icd = visit2icd.copy()
        ccs2icd = ccs2icd.copy()

        visit2icd[:,1] = visit2icd[:,1] + n_visits + n_ccss
        ccs2icd[:,1] = ccs2icd[:,1] + n_visits + n_ccss

        ccs2icd[:,0] = ccs2icd[:,0] + n_visits

        graph = (np.concatenate([row, visit2icd[:,0], ccs2icd[:,0]]), np.concatenate([col, visit2icd[:,1], ccs2icd[:,1]]))


        row, col = graph

        row_t = np.concatenate([row, col])
        col_t = np.concatenate([col, row])
        row = row_t
        col = col_t
        idx = np.unique(np.stack([row, col]), axis=1)
        vals = [1.] * (idx.shape[1])

        cf_adj = sp.coo_matrix((vals, idx), shape=(n_nodes, n_nodes))
        norm_mat = _bi_norm_lap(cf_adj)
        norm_mats = norm_mat

        return norm_mats


    def read_files(self):
        print("reading files ...")
        data_path = os.path.join(self.args.data_path, self.args.dataset)

        train_file = os.path.join(data_path, "train.txt")
        visit2icd_file = os.path.join(data_path, "visit2icd.txt")
        ccs2icd_file = os.path.join(data_path, "ccs2icd.txt")
        train_data = pd.read_csv(train_file, sep="\t")[["visitID", "ccsID"]]

        visit2icd = pd.read_csv(visit2icd_file, sep="\t")
        ccs2icd = pd.read_csv(ccs2icd_file, sep="\t")
        visit2icd, ccs2icd = visit2icd.to_numpy(), ccs2icd.to_numpy()

        train_data = train_data.to_numpy()

        test_file = os.path.join(data_path, "valid.txt")
        visit2icd_file2 = os.path.join(data_path, "visit2icd_valid.txt")
        visit2icd2 = pd.read_csv(visit2icd_file2, sep="\t")
        visit2icd2 = visit2icd2.to_numpy()
        valid_data = pd.read_csv(test_file, sep="\t")
        valid_data = valid_data.to_numpy()


        test_file = os.path.join(data_path, "test.txt")
        visit2icd_file3 = os.path.join(data_path, "visit2icd_test.txt")
        visit2icd3 = pd.read_csv(visit2icd_file3, sep="\t")
        visit2icd3 = visit2icd3.to_numpy()
        test_data = pd.read_csv(test_file, sep="\t")
        test_data = test_data.to_numpy()
        
        data_stat = self.__stat(train_data, visit2icd, ccs2icd, valid_data, test_data)

        visit2ccs_dict = defaultdict(list)
        visit2icd_dict = defaultdict(list)
        for idx in range(train_data.shape[0]):
            visit2ccs_dict[train_data[idx, 0]].append(train_data[idx, 1])

        for idx in range(visit2icd.shape[0]):
            visit2icd_dict[visit2icd[idx, 0]].append(visit2icd[idx, 1])
        data_dict = {
            "visit2ccs": visit2ccs_dict,
            "visit2icd": visit2icd_dict,
        }

        return train_data, visit2icd, ccs2icd, data_dict, data_stat, visit2icd2, valid_data, visit2icd3, test_data


    def __stat(self, train_data, visit2icd, ccs2icd, valid_data, test_data):
        n_visits = max(max(train_data[:, 0]),max(valid_data[:, 0]),max(test_data[:, 0])) + 1
        n_ccss = max(train_data[:, 1]) + 1

        data_path = os.path.join(self.args.data_path, self.args.dataset)
        icd_file = os.path.join(data_path, "icds.txt")
        icd = pd.read_csv(icd_file, sep="\t")
        icd = icd.to_numpy()
        n_icds = max(icd[:, 1]) + 1


        n_nodes = n_visits + n_ccss + n_icds

        print(f"n_visits:{n_visits}")
        print(f"n_ccss:{n_ccss}")
        print(f"n_icds:{n_icds}")
        print(f"n_nodes:{n_nodes}")
        print(f"n_interaction:{len(train_data)}")
        if visit2icd is not None:
            print(f"n_visit2icd:{len(visit2icd)}")
            print(f"n_ccs2icd:{len(ccs2icd)}")
        return {
            "n_visits": n_visits,
            "n_ccss": n_ccss,
            "n_icds": n_icds,
            "n_nodes": n_nodes
        }

