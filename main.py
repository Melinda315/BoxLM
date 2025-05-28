import torch
from utils.parser import parse_args
import numpy as np
import random
from time import time
from torch.utils.data import Dataset, DataLoader
from pyhealth.tokenizer import Tokenizer
from models.model import Model
from utils.data import DataUtils
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_init_tokenizers(task_dataset, keys = ['cond_hist']):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers


class MMDataset(Dataset):
    def __init__(self, dataset):
        self.sequence_dataset = []
        for visit in dataset.samples:
            if len(visit['cond_hist']) <= 1:
                continue
            self.sequence_dataset.append(visit)


    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        sequence_data = self.sequence_dataset[idx]
        return sequence_data, idx



def custom_collate_fn(batch):
    sequence_data_list = [item[0] for item in batch]
    graph_data_list = [item[1] for item in batch]

    sequence_data_batch = {key: [d[key] for d in sequence_data_list if d[key]!=[]] for key in sequence_data_list[0]}

    graph_data_batch = graph_data_list

    return sequence_data_batch, graph_data_batch


def batch_to_multihot(label, num_labels: int) -> torch.tensor:

    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels

def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20, 30]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = pred[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds

def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = predicts[i][:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks


def train(args):
    seed = 2025
    seed_all(seed)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    save_dir = 'logs/'+args.dataset+'.pt'


    dataUtils = DataUtils(args)
    train_data, user2tag, item2tag, data_dict, data_stat, user2tag2, valid_data, user2tag3, test_data = dataUtils.read_files()

    norm_mat = dataUtils.build_graph(train_data, user2tag, item2tag, data_stat)
    norm_mat2 = dataUtils.build_graph(valid_data, user2tag2, item2tag, data_stat)
    norm_mat3 = dataUtils.build_graph(test_data, user2tag3, item2tag, data_stat)
    
    import pickle
    task_dataset = pickle.load(open('data/'+args.dataset+'/'+args.dataset+'.pkl', 'rb'))

    df = pd.read_csv('CCSCM.csv')
    ccs = df['code'].tolist()
    label_tokenizer = Tokenizer(tokens=ccs)
    mdataset = MMDataset(task_dataset)


    from torch.utils.data import Subset
    indices = torch.load('data/'+args.dataset+'/trainset.pt')
    trainset = Subset(mdataset, indices)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    indices = torch.load('data/'+args.dataset+'/validset.pt')
    validset = Subset(mdataset, indices)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    indices = torch.load('data/'+args.dataset+'/testset.pt')
    testset = Subset(mdataset, indices)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)



    model = Model(args, data_stat, norm_mat, user2tag, item2tag, norm_mat2, norm_mat3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    early = 0
    best_epoch = 0
    bestc = None
    bestv = None
    bestv10 = 0

    
    print("start training ... ")
    for epoch in range(args.epoch):

        model.train()
        train_loss = 0
        train_s_t = time()

        for batch, (data, index) in enumerate(train_loader):
            label = prepare_labels(data['conditions'], label_tokenizer).to(device)

            embs = model.generate(mode='train')
            user_gcn_emb, entity_gcn_emb = embs
            user_gcn_emb = user_gcn_emb[index]

            rate_batch = model.rating(user_gcn_emb, entity_gcn_emb)
            criterion = torch.nn.BCELoss()
            softmax = torch.nn.Softmax()
            rate_batch = softmax(rate_batch)

            loss = criterion(rate_batch, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_e_t = time()

        print(f"Epoch {epoch}/{args.epoch} - train loss: {train_loss:.2f}")

        y_t_all, y_p_all = [], []
        with torch.no_grad():
            for batch, (data, index) in enumerate(valid_loader):
                label = prepare_labels(data['conditions'], label_tokenizer).to(device)
                y_t = label.cpu().numpy()
                y_t_all.append(y_t)

                embs = model.generate(mode='valid')
                user_gcn_emb, entity_gcn_emb = embs
                user_gcn_emb = user_gcn_emb[index]

                rate_batch = model.rating(user_gcn_emb, entity_gcn_emb).detach()
                rate_topK_val, rate_topk_idx = torch.topk(rate_batch, k=50, largest=True, dim=-1)

                y_p = rate_topk_idx.cpu().numpy()
                y_p_all.append(y_p)
            y_true = np.concatenate(y_t_all, axis=0)
            y_prob = np.concatenate(y_p_all, axis=0)

        c = code_level(y_true, y_prob)
        v = visit_level(y_true, y_prob)

        if bestv10 < v[0]:
            early = 0
            bestv10 = v[0]
            bestc = c
            bestv = v
            torch.save(model.state_dict(), save_dir)
            best_epoch = epoch
        else:
            early += 1

        valid_e_t = time()
        print('---train+val time---' + str(train_e_t - train_s_t) + '---' + str(valid_e_t - train_e_t))

        print('-----visit_level----code_level-----')
        print(v)
        print(c)

        print('----valid------best------------'+str(best_epoch))
        print(bestv)
        print(bestc)

        if early == 200:
            break

    dict_model = torch.load(save_dir)
    model.load_state_dict(dict_model)

    y_t_all, y_p_all = [], []
    with torch.no_grad():
        for batch, (data, index) in enumerate(test_loader):
            label = prepare_labels(data['conditions'], label_tokenizer).to(device)
            y_t = label.cpu().numpy()
            y_t_all.append(y_t)

            embs = model.generate(mode='test')
            user_gcn_emb, entity_gcn_emb = embs
            user_gcn_emb = user_gcn_emb[index]

            rate_batch = model.rating(user_gcn_emb, entity_gcn_emb).detach()
            rate_topK_val, rate_topk_idx = torch.topk(rate_batch, k=50, largest=True, dim=-1)

            y_p = rate_topk_idx.cpu().numpy()
            y_p_all.append(y_p)
        y_true = np.concatenate(y_t_all, axis=0)
        y_prob = np.concatenate(y_p_all, axis=0)

    c = code_level(y_true, y_prob)
    v = visit_level(y_true, y_prob)
    print('--test---visit_level----code_level-----')
    print(v)
    print(c)


if __name__ == '__main__':
    args = parse_args()
    train(args)
