import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

ccs_index = {}
icd_index = {}

f = open("ccss.txt", "r", encoding="utf-8")
data = f.readlines()[1:]
for line in data:
    line = line[:-1]
    line = line.split("\t")
    code = line[0]
    id = int(line[1])
    ccs_index[code] = id
f.close()

length = len(ccs_index)

f = open("icds.txt", "r", encoding="utf-8")
data = f.readlines()[1:]
for line in data:
    line = line[:-1]
    line = line.split("\t")
    code = line[0]
    id = int(line[1])

    icd_index[code] = id
f.close()


relations = set()

df1 = pd.read_csv('ICD9CM_to_CCSCM.csv')
childs = df1['ICD9CM'].tolist()
parents = df1['CCSCM'].tolist()

for child, parent in zip(childs, parents):
    child = child.replace('.', '')
    parent = str(parent)

    if parent in ccs_index and child in icd_index:
        relations.add((ccs_index[parent], icd_index[child]+length))


df1 = pd.read_csv('ICD9CM.csv')
childs = df1['code'].tolist()
parents = df1['parent_code'].tolist()

for child, parent in zip(childs, parents):
    if not pd.notna(parent):
        continue

    child = child.replace('.', '')
    parent = parent.replace('.', '')

    if parent in icd_index and child in icd_index:
        relations.add((icd_index[parent]+length, icd_index[child]+length))


visit_ccs = np.zeros((len(ccs_index)+len(icd_index),len(ccs_index)+len(icd_index)), dtype=int)

for parent, child in relations:
    visit_ccs[parent][child] = 1
    # visit_ccs[child][parent] = 1

coo_visit_ccs = sp.coo_matrix(visit_ccs).tocsr().astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


adj_train_norm = normalize(coo_visit_ccs)
adj_train_norm = sparse_mx_to_torch_sparse_tensor(adj_train_norm)

torch.save(adj_train_norm, 'adj-1.pt')
# torch.save(adj_train_norm, 'adj-2.pt')