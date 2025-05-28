import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min,scatter_max
from torch_scatter.composite import scatter_softmax


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings, idx, dim_size):
        layer1_act = F.relu(self.layer1(embeddings))
        layer2_act = self.layer2(layer1_act)
        attention = scatter_softmax(src=layer2_act, index=idx, dim=0)
        visit_embedding = scatter_sum(attention * embeddings, index=idx, dim=0, dim_size=dim_size)
        return visit_embedding

class BoxOffsetIntersection(nn.Module):
    def __init__(self):
        super(BoxOffsetIntersection, self).__init__()

    def forward(self, embeddings, idx, dim_size, union=False):
        if  union:
            offset = scatter_max(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        else:
            offset = scatter_min(src=embeddings, index=idx, dim=0, dim_size=dim_size)[0]
        return offset

class GraphConv(nn.Module):
    def __init__(self, emb_size, n_hops, n_visits, n_ccss, n_icds, device,
                visit2icd, ccs2icd, visit_time):
        super(GraphConv, self).__init__()

        self.n_layers = n_hops
        self.emb_size = emb_size
        self.n_visits = n_visits
        self.n_icds = n_icds
        self.n_ccss = n_ccss
        self.n_nodes = self.n_visits + self.n_ccss + self.n_icds
        self.device = device
        self.visit2icd = visit2icd
        self.ccs2icd = ccs2icd.clone()
        self.ccs2icd[:,1] = ccs2icd[:,0]
        self.ccs2icd[:,0] = ccs2icd[:,1]
        self.ccs2icd = torch.cat([ccs2icd, self.ccs2icd], dim=0)

        idx = torch.arange(self.n_visits).to(self.device)
        self.visit_union_idx = torch.cat([idx, idx, idx], dim=0)

        idx = torch.arange(self.n_ccss + self.n_icds + self.n_visits).to(self.device)
        self.all_union_idx = torch.cat([idx, idx], dim=0)


        self.center_net = CenterIntersection(self.emb_size)
        self.offset_net = BoxOffsetIntersection()

        self.visit_time = visit_time

        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)


    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def union(self, embs_list, offset_list, index):
        embs = torch.cat(embs_list, dim=0)
        offset = torch.cat(offset_list, dim=0)
        agg_emb = self.center_net(embs, index, embs_list[0].shape[0])
        ent_offset_emb = F.relu(self.offset_net(offset, index, embs_list[0].shape[0], union=True))
        
        return agg_emb, ent_offset_emb


    def forward(self, visit_emb, visit_offset, ccs_emb, ccs_offset, icd_emb, icd_offset, graph):

        _indices = graph._indices()
        head, tail = _indices[0,:], _indices[1, :]

        all_embs = torch.cat([visit_emb, ccs_emb, icd_emb], dim=0)
        all_offset_emb = F.relu(torch.cat([visit_offset, ccs_offset, icd_offset], dim=0))
        

        agg_layer_emb = [all_embs]
        agg_layer_offset = [all_offset_emb]

        time_embedding = torch.tensor(self.visit_time, dtype = torch.float32, device=self.device).view(-1,1)
        time_embedding = 1.0/time_embedding
        layer1_act = F.relu(self.layer1(time_embedding))
        layer2_act = self.layer2(layer1_act)
        time_embedding = F.softmax(layer2_act, dim=0)

        for _ in range(self.n_layers):
            entity_visit_ordinal = (head < self.n_visits) & (tail >= self.n_visits)
            history_embs = all_embs[tail[entity_visit_ordinal]]
            # icd ccs
            agg_emb1 = self.center_net(history_embs, head[entity_visit_ordinal], self.n_nodes)

            # time
            agg_emb2 = agg_emb1[:self.n_visits].clone()
            agg_emb2 = agg_emb2 * time_embedding

            # visit
            visit_visit_ordinal =  (head < self.n_visits) & (tail < self.n_visits)
            history_embs = agg_emb2[tail[visit_visit_ordinal]]
            agg_emb3 = self.center_net(history_embs, head[visit_visit_ordinal], self.n_nodes)

            agg_emb = agg_emb3

            ## visit offset
            ### visit-ccs, intersect
            inter_visit_ordinal = (head < self.n_visits) & (tail >= self.n_visits) & (tail < self.n_visits + self.n_ccss)
            
            inter_visit_history_offset = F.relu(all_offset_emb[tail[inter_visit_ordinal]])
            inter_visit_offset_emb = self.offset_net(inter_visit_history_offset, head[inter_visit_ordinal], self.n_nodes, union=True)
            inter_visit_offset_emb = inter_visit_offset_emb[:self.n_visits]
            
            ### visit-icd, union
            visit_icd_ordinal =  (head < self.n_visits) & (tail >= self.n_visits + self.n_ccss)

            visit_icd_history_offset = F.relu(all_offset_emb[tail[visit_icd_ordinal]])
            ut_visit_offset_emb = self.offset_net(visit_icd_history_offset, head[visit_icd_ordinal], self.n_nodes, union=True)
            ut_visit_offset_emb = ut_visit_offset_emb[:self.n_visits]

            ### visit-visit, union
            visit_visit_ordinal =  (head < self.n_visits) & (tail < self.n_visits)

            visit_visit_history_offset = F.relu(all_offset_emb[tail[visit_visit_ordinal]])
            visit_visit_offset_emb = self.offset_net(visit_visit_history_offset, head[visit_visit_ordinal], self.n_nodes, union=True)
            visit_visit_offset_emb = visit_visit_offset_emb[:self.n_visits]

            ### union two part 
            visit_offset = torch.cat([inter_visit_offset_emb, ut_visit_offset_emb, visit_visit_offset_emb], dim=0)
            visit_offset = F.relu(self.offset_net(visit_offset, self.visit_union_idx, inter_visit_offset_emb.shape[0], union=True))
            
            ### ccs offset
            ### intersect all neighboring nodes of ccs
            ccs_ordinal = (head >= self.n_visits) & (head < self.n_visits + self.n_ccss)

            ccs_history_offset = F.relu(all_offset_emb[tail[ccs_ordinal]])
            ccs_offset = self.offset_net(ccs_history_offset, head[ccs_ordinal], self.n_nodes, union=True)
            ccs_offset = ccs_offset[self.n_visits:self.n_visits + self.n_ccss]
            
            ### icd offset
            ### union all neighboring nodes of icd
            icd_ordinal = (head >= self.n_visits + self.n_ccss) 

            icd_history_offset = F.relu(all_offset_emb[tail[icd_ordinal]])
            icd_offset = self.offset_net(icd_history_offset, head[icd_ordinal], self.n_nodes)
            icd_offset = icd_offset[self.n_visits + self.n_ccss:]

            agg_emb = F.normalize(agg_emb)

            agg_offset_emb = F.relu(torch.cat([visit_offset, ccs_offset, icd_offset], dim=0))
            
            
            agg_layer_emb.append(agg_emb)
            agg_layer_offset.append(agg_offset_emb)
            
            all_embs = agg_emb
            all_offset_emb = agg_offset_emb

        agg_final_emb = agg_layer_emb[-1]
        agg_final_offset = agg_layer_offset[-1]

        visit_final_emb, ccs_final_emb, icd_final_emb = torch.split(agg_final_emb, [self.n_visits, self.n_ccss, self.n_icds])

        visit_final_offset, ccs_final_offset, icd_final_offset = torch.split(agg_final_offset, [self.n_visits, self.n_ccss, self.n_icds])


        return visit_final_emb, visit_final_offset


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP2, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=64):
        super(MLP, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projector(x)


class Model(nn.Module):
    def __init__(self, args, data_stat, adj_mat, visit2icd, ccs2icd, adj_mat2, adj_mat3):
        super(Model, self).__init__()

        self.beta = args.beta
        self.n_visits = data_stat['n_visits']
        self.n_ccss = data_stat['n_ccss']
        self.n_icds = data_stat["n_icds"]
        self.n_nodes = data_stat['n_nodes']

        self.emb_size = args.dim
        self.n_layers = args.layers
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

        ccs2icd[:,1] += self.n_ccss + self.n_visits
        ccs2icd[:,0] += self.n_visits
        self.visit2icd = torch.LongTensor(visit2icd).to(self.device)
        self.ccs2icd = torch.LongTensor(ccs2icd).to(self.device)
        self.adj_mat = adj_mat
        
        self._init_weight()
        self.gcn = self._init_model()

        input_dim = 768
        hidden_dim = 128
        output_dim = 16


        self.center_mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.offset_mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


        ccs_embedding = torch.load(args.data_path+'/'+args.dataset+'/ccs_embeddings.pt')
        self.ccs_embedding = torch.from_numpy(ccs_embedding).to(self.device)
        icd_embedding = torch.load(args.data_path+'/'+args.dataset+'/icd_embeddings.pt')
        self.icd_embedding = torch.from_numpy(icd_embedding).to(self.device)

        self.visit_time = torch.load(args.data_path+'/'+args.dataset+'/visit_time.pt')

        self.adj_mat2 = adj_mat2
        self.graph2 = self._convert_sp_mat_to_sp_tensor(self.adj_mat2).to(self.device)
        self.graph2 = self.add_residual(self.graph2)

        self.adj_mat3 = adj_mat3
        self.graph3 = self._convert_sp_mat_to_sp_tensor(self.adj_mat3).to(self.device)
        self.graph3 = self.add_residual(self.graph3)

        self.adj1 = torch.load(args.data_path+'/'+args.dataset+'/adj-1.pt').to(self.device)
        self.adj2 = torch.load(args.data_path+'/'+args.dataset+'/adj-2.pt').to(self.device)

        self.w1 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w1.data)
        self.w2 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w2.data)
        self.w3 = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_normal_(self.w3.data)


    def _init_model(self):
        return GraphConv(emb_size=self.emb_size,
                         n_hops=self.n_layers,
                         n_visits=self.n_visits,
                         n_ccss=self.n_ccss,
                         n_icds=self.n_icds,
                         device=self.device,
                         visit2icd=self.visit2icd,
                         ccs2icd=self.ccs2icd,
                         visit_time=self.visit_time)
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.visit_embed = initializer(torch.empty(self.n_visits, self.emb_size))
        self.visit_embed = nn.Parameter(self.visit_embed)
        self.visit_offset = initializer(torch.empty([self.n_visits, self.emb_size]))
        self.visit_offset = nn.Parameter(self.visit_offset)

        self.graph = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.graph = self.add_residual(self.graph)

    def add_residual(self, graph):
        residual_node = torch.arange(self.n_nodes).to(self.device)
        row, col = graph._indices()
        row = torch.cat([row, residual_node], dim=0)
        col = torch.cat([col, residual_node], dim=0)
        val = torch.cat([graph._values(), torch.ones_like(residual_node)])

        return torch.sparse.FloatTensor(torch.stack([row, col]), val, graph.shape).to(self.device)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def cal_logit_box(self, visit_center_embedding, visit_offset_embedding, ccs_center_embedding, ccs_offset_embedding, training=True):

        gumbel_beta = self.beta
        t1z, t1Z = visit_center_embedding - visit_offset_embedding, visit_center_embedding + visit_offset_embedding
        t2z, t2Z = ccs_center_embedding - ccs_offset_embedding, ccs_center_embedding + ccs_offset_embedding
        z = gumbel_beta * torch.logaddexp(
                t1z / gumbel_beta, t2z / gumbel_beta
            )
        z = torch.max(z, torch.max(t1z, t2z))

        Z = -gumbel_beta * torch.logaddexp(
            -t1Z / gumbel_beta, -t2Z / gumbel_beta
        )
        Z = torch.min(Z, torch.min(t1Z, t2Z))

        euler_gamma = 0.57721566490153286060

        return torch.sum(
            torch.log(
                F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=1 / gumbel_beta) + 1e-23
            ),
            dim=-1,
        )

    def lightgcn(self, ccs_center, icd_center):
        embedding = torch.cat((ccs_center, icd_center), dim=0)

        emb1 = torch.spmm(self.adj1, embedding)
        emb1 = torch.mm(emb1, self.w1)
        emb2 = torch.spmm(self.adj2, embedding)
        emb2 = torch.mm(emb2, self.w2)
        emb3 = embedding
        emb3 = torch.mm(emb3, self.w3)

        embs = emb1 * (1/3) + emb2 * (1/3) + emb3 * (1/3)

        return embs[:ccs_center.shape[0]], embs[ccs_center.shape[0]:]

    def generate(self, mode):
        if mode == 'train':
            graph = self.graph
        elif mode == 'valid':
            graph = self.graph2
        elif mode == 'test':
            graph = self.graph3

        visit_emb = self.visit_embed
        visit_offset = self.visit_offset

        ccs_emb = self.center_mlp(self.ccs_embedding)
        icd_emb = self.center_mlp(self.icd_embedding)

        ccs_offset, icd_offset = self.lightgcn(ccs_emb, icd_emb)

        visit_agg_emb, visit_agg_offset = self.gcn(visit_emb, visit_offset, ccs_emb, ccs_offset, icd_emb, icd_offset, graph)

        visit_embs = torch.cat([visit_agg_emb, visit_agg_offset], axis=-1)
        ccs_embs = torch.cat([ccs_emb, ccs_offset], axis=-1)

        return visit_embs, ccs_embs


    def rating(self, visit_embs, entity_embs):
        n_visits = visit_embs.shape[0]
        n_entities = entity_embs.shape[0]
        visit_embs = visit_embs.unsqueeze(1).expand(n_visits, n_entities,  self.emb_size * 2)
        visit_agg_embs, visit_agg_offset = torch.split(visit_embs, [self.emb_size, self.emb_size], dim=-1)

        entity_embs = entity_embs.unsqueeze(0).expand(n_visits, n_entities,  self.emb_size * 2)
        entity_agg_embs, entity_agg_offset = torch.split(entity_embs, [self.emb_size, self.emb_size], dim=-1)

        return self.cal_logit_box(visit_agg_embs, visit_agg_offset, entity_agg_embs, entity_agg_offset)
