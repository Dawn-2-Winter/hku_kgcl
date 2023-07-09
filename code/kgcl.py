import argparse
def parse_args():
  parser = argparse.ArgumentParser(description="Go lightGCN")
  parser.add_argument('--bpr_batch', type=int,default=3000, 
                        help="the batch size for bpr loss training procedure")
  parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
  parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
  parser.add_argument('--lr', type=float,default=0.0008,
                        help="the learning rate")
  parser.add_argument('--decay', type=float,default=1e-3,
                        help="the weight decay for l2 normalizaton")
  parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
  # This help info is wrong
  parser.add_argument('--keepprob', type=float,default=0.6, 
                        help="the batch size for bpr loss training procedure")
  parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
  parser.add_argument('--testbatch', type=int,default=1024,
                        help="the batch size of users for testing")
  parser.add_argument('--dataset', type=str,default='last-fm',
                        help="available datasets: [last-fm, gowalla, yelp2018, amazon-book]")
  parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
  parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
  parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
  parser.add_argument('--comment', type=str,default="lgn")
  parser.add_argument('--load', type=int,default=0)
  parser.add_argument('--epochs', type=int,default=1000)
  parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
  parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
  parser.add_argument('--seed', type=int, default=17, help='random seed')
  parser.add_argument('--model', type=str, default='kgcl', help='rec-model, support [mf, lgn, kgcl]')
  return parser.parse_args()
  
from warnings import simplefilter
import sys
import os
from os.path import join
import multiprocessing
from tqdm import tqdm
from cppimport import imp
from shutil import make_archive
import collections
import random
from time import time
from pprint import pprint
import pandas as pd
import numpy as np
from numpy import negative, positive
import torch
from torch import nn, optim, log
from torch.optim import optimizer, lr_scheduler
import torch.nn.functional as F
from torch.nn.init import ones_
from torch_sparse.tensor import to
from torch_geometric.utils import degree, to_undirected
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
import scipy.sparse as sp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
# Need to change this!!!!!!!!!!!!!!!!
ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
  os.makedirs(FILE_PATH, exist_ok=True)

config = {}
all_dataset = ['last-fm', 'MIND']
all_models = ['kgcl']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

kgcn = 'RGAT'
train_trans = False
entity_num_per_item = 10
uicontrast = "RANDOM"# RANDOM
kgc_enable = True
kgc_joint = True
kgc_temp = 0.1 # 0.2
use_kgc_pretrain = False
pretrain_kgc = False
kg_p_drop = 0.5
ui_p_drop = 0.0001
ssl_reg = 0.1
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
test_verbose = 1
test_start_epoch = 1
early_stop_cnt = 10
#mix_ratio = 1-ui_p_drop-0

dataset = args.dataset
if dataset == 'MIND':
  config['lr'] = 5e-4
  config['decay'] = 1e-3
  config['dropout'] = 1
  config['keep_prob'] = 0.6

  uicontrast = "WEIGHTED-MIX"
  kgc_enable = True
  kgc_joint = True
  use_kgc_pretrain = False
  entity_num_per_item = 6
  # [0.06, 0.08, 0.1]
  ssl_reg = 0.06
  kgc_temp = 0.2
  # [0.3, 0.5, 0.7]
  kg_p_drop = 0.5
  # [0.1, 0.2, 0.4]
  ui_p_drop = 0.4
  mix_ratio = 1-ui_p_drop-0
  test_start_epoch = 1
  early_stop_cnt = 3


TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
simplefilter(action="ignore", category=FutureWarning)
model_name = 'kgcl'
def cprint(words: str):
  print(words)

class BasicDataset(Dataset):
  def __init__(self):
    print("init dataset")
  
  @property
  def n_users(self):
    raise NotImplementedError
  
  @property
  def m_items(self):
    raise NotImplementedError
  
  @property
  def trainDataSize(self):
    raise NotImplementedError
  
  @property
  def testDict(self):
    raise NotImplementedError
  
  @property
  def allPos(self):
    raise NotImplementedError
  
  def getUserItemFeedback(self, users, items):
    raise NotImplementedError
  
  def getUserPosItems(self, users):
    raise NotImplementedError
  
  def getUserNegItems(self, users):
    """
    not necessary for large dataset
    it's stupid to return all neg items in super large dataset
    """
    raise NotImplementedError
  
  def getSparseGraph(self):
    """
    build a graph in torch.sparse.IntTensor.
    Details in NGCF's matrix form
    A = 
        |I,   R|
        |R^T, I|
    """
    raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    # # Need to change!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, config = config, path="./data/last-fm"): # Need to change!!
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']      # False
        self.folds = config['A_n_fold']     # 100
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train'] # 0
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]] # [6632 28210 17214 5644 2370 7963 21147]
                    uid = int(l[0])                 # 0
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items)) # [0 0 0 0 0 0 0...]
                    trainItem.extend(items)              # [6632 28210 17214 5644 2370 7963 21147...]
                    self.m_item = max(self.m_item, max(items)) # item数量
                    self.n_user = max(self.n_user, uid)        # user数量
                    self.traindataSize += len(items)           # triplet数量
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1]:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.validDataSize += len(items)
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)
        
        self.m_item += 1
        self.n_user += 1
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph # csr_matrix((data, indices, indptr))
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                # E^(k+1)=(D^(-1/2) A D^(-1/2)) E^(k) # A是每行的非零个数组成的 # E是embedding矩阵
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems

    # train loader and sampler part
    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg

class KGDataset(Dataset):
    def __init__(self, kg_path=join(DATA_PATH, dataset, "kg.txt")):
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data(kg_data=self.kg_data)
        self.item_net_path = join(DATA_PATH, dataset)

    @property
    def entity_count(self):
        # start from zero
        return self.kg_data['t'].max()+2

    @property
    def relation_count(self):
        return self.kg_data['r'].max()+2

    def get_kg_dict(self, item_num):
        entity_num = entity_num_per_item # 6
        i2es = dict()
        i2rs = dict()
        for item in range(item_num):
            rts = self.kg_dict.get(item, False)
            if rts:
                tails = list(map(lambda x:x[1], rts)) # 截断
                relations = list(map(lambda x:x[0], rts))
                if(len(tails) > entity_num):
                    i2es[item] = torch.IntTensor(tails).to(device)[:entity_num]
                    i2rs[item] = torch.IntTensor(relations).to(device)[:entity_num]
                else:
                    # last embedding pos as padding idx
                    tails.extend([self.entity_count]*(entity_num-len(tails))) # 填充
                    relations.extend([self.relation_count]*(entity_num-len(relations)))
                    i2es[item] = torch.IntTensor(tails).to(device)
                    i2rs[item] = torch.IntTensor(relations).to(device)
            else: # 没有connection就全部填充
                i2es[item] = torch.IntTensor([self.entity_count]*entity_num).to(device)
                i2rs[item] = torch.IntTensor([self.relation_count]*entity_num).to(device)
        return i2es, i2rs # 返回的是一个item与不同的entities和relation的Tensor


    def generate_kg_data(self, kg_data): 
        # construct kg dict
        kg_dict = collections.defaultdict(list)
        for row in kg_data.iterrows():
            h, r, t = row[1]    # ? Why row[1]
            kg_dict[h].append((r, t))
        heads = list(kg_dict.keys())
        return kg_dict, heads

    def __len__(self):
        return len(self.kg_dict)

    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head]) # 随机选一个head的relation
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail # 找当前head的正connection，pos_tail以及一个neg_tail
 
 
if dataset in ['movielens', 'last-fm', 'MIND', 'yelp2018', 'amazon-book']:
  load_dataset = Loader(path=join(DATA_PATH, dataset))
  kg_dataset = KGDataset()
#elif dataset == 'last-fm':
  #load_dataset = LastFM(path=join(DATA_PATH, dataset))
  #kg_dataset = KGDataset()
print('===========config================')
print(f"PID: {os.getpid()}")
print("KGCN:{}, TransR:{}, N:{}".format(kgcn, train_trans, entity_num_per_item))
print("KGC: {} @ d_prob:{} @ joint:{} @ from_pretrain:{}".format(kgc_enable, kg_p_drop, kgc_joint, use_kgc_pretrain))
print("UIC: {} @ d_prob:{} @ temp:{} @ reg:{}".format(uicontrast, ui_p_drop, kgc_temp, ssl_reg))
pprint(config)
print("cores for test:", CORES)
print("comment:", comment)
print("tensorboard:", tensorboard)
print("LOAD:", LOAD)
print("Weight path:", PATH)
print("Test Topks:", topks)
print("using bpr loss")
print('===========end===================')

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha): # 64, 64, 0.4, 0.2
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) # 64, 64, 0.4, 0.2, False

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj) # update item_embs using GAT
        x = F.dropout(x, self.dropout, training=self.training)
        return x



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True): # 64, 64, 0.4, 0.2, False
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout           # 0.4
        self.in_features = in_features   # 64
        self.out_features = out_features # 64
        self.alpha = alpha               # 0.2
        self.concat = concat             # False

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) # self.W.size: 64 x 64
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))         # self.a.size: 128 x 1
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*out_features, out_features)                    # self.fc: 128 x 64

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    # Equ 1 in paper
    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        
        # N, e_num, dim
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime # This way

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W) # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We) # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size()) # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1) # (N, e_num, 2*out_features)


def _L2_loss_mean(x):
  return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class KGCL(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset,
                 kg_dataset):
        super(KGCL, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))

        self.latent_dim = self.config['latent_dim_rec']  # args.recdim   6 4
        self.n_layers = self.config['lightGCN_n_layers'] # args.layer     3
        self.keep_prob = self.config['keep_prob']        # args.keepprob 0.8
        self.A_split = self.config['A_split']            # False

        # User embedding
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim) # 300000 x 64
        # Item embedding
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim) # 48957 x 64
        # Entity embedding
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim) # 57474 x 64
        # Relation embedding
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim) # 64 x 64

        # relation weights
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim)) # 63 x 64 x 64
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config['pretrain'] == 0:
            cprint('use NORMAL distribution UI')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution ENTITY')
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        print(f"KGCL is ready to go!")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def view_computer_all(self, g_droped, kg_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer(self): 
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict) # update the items_emb in equ 1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)# 3
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer() # 2
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long()) # 1
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(
            torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if (torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)                 # (kg_batch_size, relation_dim)
        h_embed = self.embedding_item(h)
        pos_t_embed = self.embedding_entity(pos_t)           # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)           # (kg_batch_size, entity_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)
        # torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)
        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(
            r)                 # (kg_batch_size, relation_dim)
        # (kg_batch_size, entity_dim, relation_dim)
        W_r = self.W_R[r]

        # (kg_batch_size, entity_dim)
        h_embed = self.embedding_item(h)
        pos_t_embed = self.embedding_entity(
            pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(
            neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(
            1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(
            1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(
            1)     # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + \
            _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    def cal_item_embedding_gat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).to(device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        return self.gat(item_embs, entity_embs, padding_mask)

    def cal_item_embedding_rgat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        if kgcn == "GAT":
            return self.cal_item_embedding_gat(kg)
        elif kgcn == "RGAT":
            return self.cal_item_embedding_rgat(kg)
        elif (kgcn == "MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif (kgcn == "NO"):
            return self.embedding_item.weight

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(
            list(kg.keys())).to(device))  # item_num, emb_dim
        # item_num, entity_num_each
        item_entities = torch.stack(list(kg.values()))
        # item_num, entity_num_each, emb_dim
        entity_embs = self.embedding_entity(item_entities)
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(
            item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * \
            padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / \
            padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs+entity_embs_mean

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
       
       
       
       
def drop_edge_random(item2entities, p_drop, padding): # kg = item:[entity1, entity2, ...], kg_p_drop = 0.5, self.gcn_model.num_entities是entity--t的个数(h,r,t)
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if (random.random() > p_drop):
                new_es.append(e)
            else:
                new_es.append(padding)
        res[item] = torch.IntTensor(new_es).to(device)
    return res
# len(self.gcn_model.dataset.trainUser), size=int(len(self.gcn_model.dataset.trainUser) * (1 - p_drop)), replace=False)
def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample

class Contrast(nn.Module):
    def __init__(self, gcn_model, tau=kgc_temp):
        super(Contrast, self).__init__()
        self.gcn_model: KGCL = gcn_model
        self.tau = tau # 0.2

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss_overall(self, z1, z2, z_all):
        def f(x): return torch.exp(x / self.tau)
        # batch_size
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def get_kg_views(self):
        kg = self.gcn_model.kg_dict # item:[entity1, entity2, ...]
        view1 = drop_edge_random(kg, kg_p_drop, self.gcn_model.num_entities) # kg_p_drop = 0.5, self.gcn_model.num_entities是entity t的个数
        view2 = drop_edge_random(kg, kg_p_drop, self.gcn_model.num_entities)
        return view1, view2

    def get_ui_views_weighted(self, item_stabilities):
        graph = self.gcn_model.Graph
        n_users = self.gcn_model.num_users

        # generate mask
        item_degrees = degree(graph.indices()[0])[n_users:].tolist()
        deg_col = torch.FloatTensor(item_degrees).to(device)
        s_col = torch.log(deg_col)
        # degree normalization
        # deg probability of keep
        degree_weights = (s_col - s_col.min()) / (s_col.max() - s_col.min())
        degree_weights = degree_weights.where(
            degree_weights > 0.3, torch.ones_like(degree_weights) * 0.3)  # p_tau

        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / \
            (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(
            kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = (1-ui_p_drop)/torch.mean(kg_weights)*(kg_weights)
        weights = weights.where(
            weights < 0.95, torch.ones_like(weights) * 0.95)

        item_mask = torch.bernoulli(weights).to(torch.bool)
        print(f"keep ratio: {item_mask.sum()/item_mask.size()[0]:.2f}")
        # drop
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, view2):
        kgv1_ro = self.gcn_model.cal_item_embedding_from_kg(view1)
        kgv2_ro = self.gcn_model.cal_item_embedding_from_kg(view2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim

    def ui_drop_weighted(self, item_mask):
        # item_mask: [item_num]
        item_mask = item_mask.tolist()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        # [interaction_num]
        item_np = self.gcn_model.dataset.trainItem
        keep_idx = list()
        if uicontrast == "WEIGHTED-MIX":
            # overall sample rate = 0.4*0.9 = 0.36
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j] and random.random() > 0.6:
                    keep_idx.append(i)
            # add random samples
            interaction_random_sample = random.sample(list(range(len(item_np))), int(len(item_np)*mix_ratio))
            keep_idx = list(set(keep_idx+interaction_random_sample))
        else:
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j]:
                    keep_idx.append(i)

        print(f"finally keep ratio: {len(keep_idx)/len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        user_np = self.gcn_model.dataset.trainUser[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(device)
        g.requires_grad = False
        return g
    # 
    def ui_drop_random(self, p_drop): # p_drop = 0.4
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        # 以p的概率keep一些idx
        keep_idx = randint_choice(len(self.gcn_model.dataset.trainUser), size=int(len(self.gcn_model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = self.gcn_model.dataset.trainUser[keep_idx]
        item_np = self.gcn_model.dataset.trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        # csr_matrix((data, indices, indptr), shape=(3, 3))
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col]) # 升维度的stack
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(device)
        g.requires_grad = False
        return g

    def get_views(self, aug_side="both"):
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        if aug_side == "ui" or not kgc_enable:
            kgv1, kgv2 = None, None

        else: # This way
            kgv1, kgv2 = self.get_kg_views() # 按照0.5的p，用padding来替换初始entity——t

        if aug_side == "kg" or uicontrast == "NO" or uicontrast == "ITEM-BI":
            uiv1, uiv2 = None, None
        else:
            if uicontrast == "WEIGHTED" or uicontrast == "WEIGHTED-MIX": # This way
                # [item_num]
                stability = self.item_kg_stability(kgv1, kgv2).to(device)
                # item drop -> 2 views
                uiv1 = self.get_ui_views_weighted(stability)
                uiv2 = self.get_ui_views_weighted(stability)

            elif uicontrast == "RANDOM":
                uiv1 = self.ui_drop_random(ui_p_drop)
                uiv2 = self.ui_drop_random(ui_p_drop)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views
  

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

def TransR_train(recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()
    kgdataset = KGDataset()
    kgloader = DataLoader(kgdataset, batch_size=4096, drop_last=True) # kgloader[0] = head, relation, pos_tail, neg_tail
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True): # kgloader[0] = head, relation, pos_tail, neg_tail
        heads = data[0].to(device)
        relations = data[1].to(device)
        pos_tails = data[2].to(device)
        neg_tails = data[3].to(device)

        kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()

# neg_k=1, w=None
def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model: Contrast, contrast_views, epoch, optimizer, neg_k=1, w=None):
    Recmodel: KGCL = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    batch_size = config['bpr_batch_size']
    # __getitem__返回user, positem, negitem
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    total_batch = len(dataloader)
    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.
    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=True): # [user, positem, negitem]
        batch_users = train_data[0].long().to(device)
        batch_pos = train_data[1].long().to(device)
        batch_neg = train_data[2].long().to(device)

        # main task (batch based)
        # bpr loss for a batch of users
        l_main = bpr.compute(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos  # [B*1]

        if uicontrast != "NO":
            # do SGL:
            # readout
            if kgc_joint:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, kgv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, kgv2)
            else:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_ui(uiv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_ui(uiv2)
            # from SGL source
            items_uiv1 = itemsv1_ro[items]
            items_uiv2 = itemsv2_ro[items]
            l_item = contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)

            users = batch_users
            users_uiv1 = usersv1_ro[users]
            users_uiv2 = usersv2_ro[users]
            l_user = contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)
            # l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
            # L = L_main + L_user + L_item + L_kg + R^2
            l_ssl.extend([l_user*ssl_reg, l_item*ssl_reg]) # ssl_reg = 0.06

        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main+l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        if tensorboard:
            w.add_scalar(f'BPRLoss/BPR', l_all, epoch *
                         int(len(users) / config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}"


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    S = UniformSample_original_python(dataset)
    return S
  
def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class

    with timer(name="Main"):
        S = UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch = len(users) // config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch *
                         int(len(users) / config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = config['test_u_batch_size']
    dataset: BasicDataset
    testDict: dict = dataset.testDict
    Recmodel:KGCL
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if tensorboard:
            w.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
        
class BPRLoss:
    def __init__(self,
                 recmodel,
                 opt):
        self.model = recmodel
        self.opt = opt
        self.weight_decay = config["decay"] # 1e-4

    def compute(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        return loss

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def getFileName():
    if model_name == 'mf':
        file = f"mf-{dataset}-{config['latent_dim_rec']}.pth.tar"
    elif model_name == 'lgn':
        file = f"lgn-{dataset}-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    elif model_name == 'kgcl':
        file = f"kgc-{dataset}-{config['latent_dim_rec']}.pth.tar"
    elif model_name == 'sgl':
        file = f"sgl-{dataset}-{config['latent_dim_rec']}.pth.tar"
    return os.path.join(FILE_PATH, file)
    
    
def set_seed(seed):
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.manual_seed(seed)

set_seed(seed)
print(">>SEED:", seed)
Recmodel = KGCL(config, load_dataset, kg_dataset)
#Recmodel.load_state_dict(torch.load('/userhome/cs2/cdylxslj/anaconda3/envs/Object_Detection/KGCL/code/checkpoints/kgc-last-fm-64.pth.tar'))
Recmodel = Recmodel.to(device)
contrast_model = Contrast(Recmodel).to(device)
optimizer = optim.Adam(Recmodel.parameters(), lr=config['lr'])
if dataset == "MIND":
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.2)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)
bpr = BPRLoss(Recmodel, optimizer) 
weight_file = getFileName()
print(f"load and save to {weight_file}")

if LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1
# init tensorboard
if tensorboard:
    w : SummaryWriter = SummaryWriter(join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + comment))
else:
    w = None
    cprint("not enable tensorflowboard")

try:
    # for early stop
    # recall@20
    least_loss = 1e5
    best_result = 0.
    stopping_step = 0

    for epoch in tqdm(range(TRAIN_epochs), disable=True):
        start = time()
        # transR learning
        if epoch%1 == 0:
            if train_trans: # True
                cprint("[Trans]")
                trans_loss = TransR_train(Recmodel, optimizer)
                print(f"trans Loss: {trans_loss:.3f}")

        
        # joint learning part
        if not pretrain_kgc:
            cprint("[Drop]")
            if kgc_joint: # True
                contrast_views = contrast_model.get_views()
                # contrast_views:
                # "kgv1": kgv1,
                # "kgv2": kgv2,
                # "uiv1": uiv1,
                # "uiv2": uiv2
            else:
                contrast_views = contrast_model.get_views("ui")
            
            cprint("[Joint Learning]")
            if  kgc_joint or uicontrast!="NO": # This Way
                output_information = BPR_train_contrast(load_dataset, Recmodel, bpr, contrast_model, contrast_views, epoch, optimizer, neg_k = Neg_k, w = w) # neg_k=1, w=None
            else:
                # LightGCN
                output_information = BPR_train_original(load_dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            

            print(f'EPOCH[{epoch+1}/{TRAIN_epochs}] {output_information}')

            if epoch < test_start_epoch:
                if epoch %5 == 0:
                    cprint("[TEST]")
                    Test(load_dataset, Recmodel, epoch, w, config['multicore'])
            else:
                if epoch % test_verbose == 0:
                    cprint("[TEST]")
                    result = Test(load_dataset, Recmodel, epoch, w, config['multicore'])
                    if result["recall"] > best_result:
                        stopping_step = 0
                        best_result = result["recall"]
                        print("find a better model")
                        print('The result is:')
                        pprint(result)
                        torch.save(Recmodel.state_dict(), weight_file)
                    else:
                        stopping_step += 1
                        if stopping_step >= early_stop_cnt:
                            print(f"early stop triggerd at epoch {epoch}")
                            break
        
        scheduler.step()
finally:
    if tensorboard:
        w.close()















