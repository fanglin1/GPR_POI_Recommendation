import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from time import time
import scipy.sparse as sp
import torch
from tqdm import tqdm
import torch.nn as nn
import os


class Loader():
    def __init__(self, dataset_name='Gowalla', path='./dataset/', save_path='./data_save/distance_graph'):

        path = path + dataset_name + "/"
        save_path = save_path + dataset_name + '.npy'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        size_path = path + dataset_name + '_data_size.txt'
        self.check_in_file = path + dataset_name + '_checkins.txt'
        self.poi_geo_file = path + dataset_name + '_poi_coos.txt'
        train_file = path + dataset_name + '_train.txt'
        test_file = path + dataset_name + '_test.txt'
        val_file = path + dataset_name + '_tune.txt'
        self.save_path = save_path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        valUniqueUsers, valItem, valUser = [], [], []
        with open(size_path) as f:
            size_lis = f.readline().strip('\n').split('\t')
            self.user_num = int(size_lis[0])
            self.poi_num = int(size_lis[1])
        # 读取数据
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid = int(l[0])
                    poi = int(l[1])
                    freq = int(l[2])
                    if uid not in trainUniqueUsers:
                        trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * int(freq))
                    trainItem.extend([poi] * int(freq))
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainSize = len(self.trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid = int(l[0])
                    poi = int(l[1])
                    freq = int(l[2])
                    if uid not in testUniqueUsers:
                        testUniqueUsers.append(uid)
                    testUser.extend([uid] * int(freq))
                    testItem.extend([poi] * int(freq))
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid = int(l[0])
                    poi = int(l[1])
                    freq = int(l[2])
                    if uid not in valUniqueUsers:
                        valUniqueUsers.append(uid)
                    valUser.extend([uid] * int(freq))
                    valItem.extend([poi] * int(freq))
        self.valUniqueUsers = np.array(valUniqueUsers)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        print('数据读取完成，开始构建图')
        # 用户-poi的无向图，值为1
        self.UserPoiNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                     shape=(self.user_num, self.poi_num))

        self.sparse_user_poi_graph = self.get_graph().to(self.device)
        print('用户-POI图构建完成，开始构建训练集、测试集、验证集')

        #  二维列表，列表中每个元素为为一个一维列表，存储一个用户访问过的poi
        self.user_pois = self.getUserPois(list(range(self.user_num)))
        self.test_data = self.build_test(testItem, testUser)
        self.val_data = self.build_test(valItem, valUser)
        # self.graph = self.get_graph()
        print('开始构建有向图及POI距离矩阵')
        train_check_in = self.split_data()

        self.directed_poi_graph, self.directed_in_graph, self.target_graph = self.get_poi_graph(train_check_in)
        if os.path.exists(save_path):
            is_saved = True
        else:
            #             os.mkdir('./data_save')
            is_saved = False
        if is_saved:
            print('开始加载距离图矩阵')
            self.distance_graph = torch.FloatTensor(np.load(self.save_path)).to(self.device)
        else:
            print('开始计算距离图矩阵')
            self.distance_graph = self.calculate_poi_distance(saved=is_saved)
        print('数据初始化完成')

    # 获取用户访问过的记录
    def getUserPois(self, users):
        posItems = []
        for user in users:
            # print(user,self.UserItemNet[user].shape)
            posItems.append(self.UserPoiNet[user].nonzero()[1])
        return posItems

    # 获取测试集与验证集
    def build_test(self, poi_data, user_data):

        test_data = {}
        for i, poi in enumerate(poi_data):
            user = user_data[i]
            if test_data.get(user):
                test_data[user].append(poi)
            else:
                test_data[user] = [poi]
        return test_data

    # 建立无向图
    def get_graph(self):
        adj_mat = sp.dok_matrix((self.user_num, self.poi_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserPoiNet.tolil()
        adj_mat[:self.user_num, :self.poi_num] = R

        adj_mat = adj_mat.tocsr()

        Graph = self._convert_sp_mat_to_sp_tensor(adj_mat)

        return Graph

    # 转换稀疏矩阵为tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    # 距离计算
    def haversine_distances_fast(self, X, Y):

        dlon = Y[:, 1] - X[:, 1][:, np.newaxis]
        dlat = Y[:, 0] - X[:, 0][:, np.newaxis]
        a = np.sin(dlat / 2) ** 2 + np.cos(X[:, 0])[:, np.newaxis] * np.cos(Y[:, 0]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    # 如果已保存了距离文件直接加载即可
    def calculate_poi_distance(self, saved=False):
        geo = \
            pd.read_csv(self.poi_geo_file, header=None, sep='\t', names=['poi_id', 'lat', 'lon']).sort_values(
                by='poi_id',
                ascending=True)[
                ['lat', 'lon']].astype(np.float32)

        lat_rad = np.radians(geo.loc[:, 'lat'].values)
        lon_rad = np.radians(geo.loc[:, 'lon'].values)
        undisturbed_lat_rad = np.radians(geo.loc[:, 'lat'].values)
        undisturbed_lon_rad = np.radians(geo.loc[:, 'lon'].values)

        dist_matrix = self.haversine_distances_fast(np.vstack([lat_rad, lon_rad]).T,
                                                    np.vstack([undisturbed_lat_rad, undisturbed_lon_rad]).T)
        dist_matrix = np.maximum(dist_matrix, 0.01)
        dist_matrix = np.minimum(dist_matrix, 100)

        if not saved:
            np.save(self.save_path, dist_matrix)
        return torch.FloatTensor(dist_matrix).to(self.device)

    # 划分原始数据
    def split_data(self):
        data_df1 = pd.read_csv(self.check_in_file, sep='\t', header=None, names=['user_id', 'poi_id', 'time'])
        data_df1['time'] = pd.to_datetime(data_df1['time'], unit='s')
        users = data_df1['user_id'].unique()
        total_train = []

        for user in tqdm(users):
            user_df = data_df1[data_df1['user_id'] == user].sort_values(['time'], ascending=[True]).reset_index(
                drop=True)
            user_data_num = user_df.shape[0]
            # 用户的前百分之70为测试集中数据
            train_df = user_df[:int(user_data_num * 0.7)]
            total_train.append(train_df)
        total_train = pd.concat(total_train, axis=0).reset_index(drop=True)
        return total_train

    # 获取有向图
    def get_poi_graph(self, data_train):
        graph = np.zeros((self.poi_num, self.poi_num))
        users = data_train['user_id'].unique()
        for user in tqdm(users):
            user_data = data_train.loc[data_train['user_id'] == user].sort_values('time', ascending=True).reset_index(
                drop=True)
            index = user_data.index
            for i in range(1, len(index)):
                # 计算时间间隔
                time_interval = user_data.loc[index[i], 'time'] - user_data.loc[index[i - 1], 'time']
                if time_interval.days < 1:
                    poi1 = user_data.loc[index[i - 1], 'poi_id']
                    poi2 = user_data.loc[index[i], 'poi_id']
                    if poi1 != poi2:
                        graph[poi1, poi2] += 1
        # 结点的度
        D1 = graph.sum(axis=1).reshape(-1, 1)
        D1[D1 == 0] = 1
        print('图的维度', graph.shape, '度的维度', D1.shape)
        in_graph = graph.T
        D2 = in_graph.sum(axis=1).reshape(-1, 1)
        D2[D2 == 0] = 1
        norm_graph = self._convert_sp_mat_to_sp_tensor(csr_matrix(graph / D1)).coalesce().to(self.device)
        norm_in_graph = self._convert_sp_mat_to_sp_tensor(csr_matrix(in_graph / D2)).coalesce().to(self.device)

        return norm_graph, norm_in_graph, torch.FloatTensor(graph).to(self.device)


loader = Loader()

import torch.nn.functional as F


class GGLR(nn.Module):
    def __init__(self, d, out_graph, in_graph):
        super(GGLR, self).__init__()
        self.out_weight = nn.Parameter(torch.FloatTensor(d, d))
        self.in_weight = nn.Parameter(torch.FloatTensor(d, d))
        self.bias1 = nn.Parameter(torch.FloatTensor(d))
        self.bias2 = nn.Parameter(torch.FloatTensor(d))
        self.out_g = out_graph
        self.in_g = in_graph

    def forward(self, x1, x2):
        # print(x1.dtype,x2.dtype,self.out_g.dtype,self.out_weight.dtype)

        # assert not torch.any(torch.isnan(self.out_g))
        # assert not torch.any(torch.isnan(self.in_g))
        # assert not torch.any(torch.isnan(x1))
        # assert not torch.any(torch.isnan(x2))
        # print('x1\n',x1)
        # print('out_weight\n', self.out_weight)
        # print(torch.mm(x1,self.out_weight))
        # print(torch.sparse.mm(self.out_g,torch.mm(x1,self.out_weight)))
        # assert not torch.any(torch.isnan(torch.mm(x1,self.out_weight)))
        # assert not torch.any(torch.sparse.mm(self.out_g,torch.mm(x1,self.out_weight)))

        k_emb_outgoing = torch.sparse.mm(self.out_g, torch.mm(x1, self.out_weight)) + self.bias1
        k_emb_ingoing = torch.sparse.mm(self.in_g, torch.mm(x2, self.in_weight)) + self.bias2
        return F.relu(k_emb_outgoing), F.relu(k_emb_ingoing)


class User(nn.Module):
    def __init__(self, dim, user_poi):
        super(User, self).__init__()
        self.user_weight = nn.Parameter(torch.FloatTensor(dim, dim))
        self.poi_weight = nn.Parameter(torch.FloatTensor(dim, dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.adj = user_poi

    def forward(self, all_out_going_embs, user_embs, selected_u):
        selected_users_adj = self.adj.index_select(0, selected_u)
        poi_message = torch.sparse.mm(selected_users_adj, torch.mm(all_out_going_embs, self.poi_weight))
        user_message = torch.mm(user_embs, self.user_weight)
        user_update_embs = poi_message + user_message + self.bias
        return F.relu(user_update_embs)


class GPR(nn.Module):
    def __init__(self, data, latent_dim, layer_num):
        super(GPR, self).__init__()
        self.user_num = data.user_num
        self.poi_num = data.poi_num
        self.dim = latent_dim
        self.layer_num = layer_num
        self.adj = data.sparse_user_poi_graph.to(loader.device)
        self.target_adj = data.target_graph

        # 出度图和入度图
        self.out_going_adj = data.directed_poi_graph
        self.in_going_adj = data.directed_in_graph
        # poi更新
        self.poi_module = nn.ModuleList(
            [GGLR(self.dim, self.out_going_adj, self.in_going_adj) for _ in range(self.layer_num)])
        # 用户更新
        self.user_module = nn.ModuleList([User(self.dim, self.adj) for _ in range(self.layer_num)])

        # 用户和poi的embedding
        self.user_embedding = nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.dim)
        self.ingoing_poi_embedding = nn.Embedding(num_embeddings=self.poi_num, embedding_dim=self.dim)
        self.outgoing_poi_embedding = nn.Embedding(num_embeddings=self.poi_num, embedding_dim=self.dim)

        # 距离参数
        self.a = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.c = nn.Parameter(torch.FloatTensor(1))
        self.distance_graph = data.distance_graph
        self.distance_weight = nn.Parameter(torch.FloatTensor(self.dim, self.dim))

    def bpr_loss(self, users_emb, pos_emb, neg_emb):

        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users_emb))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def mean_square_loss(self, pre, labels):
        # print('pre.shape',pre.shape,labels.shape)
        loss_fun = torch.nn.L1Loss()
        return loss_fun(pre, labels)

    def test_rating(self, users):
        all_user_embs = self.user_embedding.weight
        all_ingoing_poi_embs = self.ingoing_poi_embedding.weight
        all_outgoing_poi_embs = self.outgoing_poi_embedding.weight
        selected_users = all_user_embs[users]
        all_layer_user_embs = []
        all_layer_in_embs = []
        for i in range(self.layer_num):
            all_outgoing_poi_embs, all_ingoing_poi_embs = self.poi_module[i](all_outgoing_poi_embs,
                                                                             all_ingoing_poi_embs)
            selected_users = self.user_module[i](all_outgoing_poi_embs, selected_users, users)
            all_layer_user_embs.append(selected_users)
            all_layer_in_embs.append(all_ingoing_poi_embs)

        concat_user = torch.concat(all_layer_user_embs, 1)

        concat_in = torch.concat(all_layer_in_embs, 1)
        user_scores = torch.matmul(concat_user, concat_in.T)
        return user_scores

    def forward(self, users, pos_poi, neg_poi):
        # print('users',users,'pos_poi',pos_poi,'neg_poi',neg_poi)
        all_user_embs = self.user_embedding.weight
        all_ingoing_poi_embs = self.ingoing_poi_embedding.weight
        all_outgoing_poi_embs = self.outgoing_poi_embedding.weight
        # print('all_user_embs\n',all_user_embs)
        # print('users\n',users)
        selected_users = all_user_embs[users]
        # print('selected_users.shape',selected_users.shape)

        all_layer_user_embs = []
        all_layer_out_embs = []
        all_layer_in_embs = []
        for i in range(self.layer_num):
            all_outgoing_poi_embs, all_ingoing_poi_embs = self.poi_module[i](all_outgoing_poi_embs,
                                                                             all_ingoing_poi_embs)
            #             print(all_outgoing_poi_embs.deivice,selected_users.deivice,users.device)
            #             print('assert not torch.any(torch.isnan(all_outgoing_poi_embs))')
            #             assert not torch.any(torch.isnan(all_outgoing_poi_embs))
            #             # print('assert not torch.any(torch.isnan(all_ingoing_poi_embs))')
            #             assert not torch.any(torch.isnan(all_ingoing_poi_embs))

            selected_users = self.user_module[i](all_outgoing_poi_embs, selected_users, users)
            # print('assert not torch.any(torch.isnan(selected_users))')
            # assert not torch.any(torch.isnan(selected_users))
            all_layer_user_embs.append(selected_users)
            all_layer_out_embs.append(all_outgoing_poi_embs)
            all_layer_in_embs.append(all_ingoing_poi_embs)

        # 物理地理影响
        physical_geography_effect = self.a * (self.distance_graph.pow(self.b)) * torch.exp(self.c * self.distance_graph)
        physical_geography_effect = torch.where(physical_geography_effect > 1e9, 1e9, physical_geography_effect)
        physical_geography_effect = torch.where(physical_geography_effect < -1e9, -1e9, physical_geography_effect)
        # assert not torch.any(torch.isnan(physical_geography_effect))
        # print('physical_geography_effect',physical_geography_effect,'\nmax',torch.max(physical_geography_effect),'min',torch.min(physical_geography_effect),'a',self.a,'b',self.b,'c',self.c)
        # 频次预测值
        freq = physical_geography_effect * (
            torch.mm(torch.mm(all_outgoing_poi_embs, self.distance_weight), all_ingoing_poi_embs.T))

        mse_loss = self.mean_square_loss(freq, self.target_adj)
        # 拼接
        concat_user = torch.concat(all_layer_user_embs, 1)
        concat_out = torch.concat(all_layer_out_embs, 1)
        concat_in = torch.concat(all_layer_in_embs, 1)
        # 被选的用户
        concat_in_pos = concat_in[pos_poi]
        concat_in_neg = concat_in[neg_poi]
        bpr_loss, bpr_reg_loss = self.bpr_loss(concat_user, concat_in_pos, concat_in_neg)

        # 用户评分
        user_scores = torch.matmul(concat_user, concat_in.T)
        # print(mse_loss, bpr_loss, bpr_reg_loss)

        return user_scores, mse_loss, bpr_loss, bpr_reg_loss


from torch import optim
import multiprocessing
import torch


def get_pos_neg_data(dataset):
    all_data = dataset.user_pois
    users = np.random.randint(0, dataset.user_num, dataset.trainSize)
    samples = []
    for ind, user in enumerate(users):
        user_pois = all_data[user]
        if len(user_pois) == 0:
            continue
        poi_index = np.random.randint(0, len(user_pois))
        # 正样本
        pos_poi_id = user_pois[poi_index]
        # 负样本
        while True:
            neg_poi_id = np.random.randint(0, dataset.poi_num)
            if neg_poi_id in user_pois:
                continue
            else:
                break
        samples.append((user, pos_poi_id, neg_poi_id))
    return np.array(samples)


def shuffle_data(*args):
    shuffle_indices = np.arange(len(args[0]))
    if len(set(len(x) for x in args)) != 1:
        raise ValueError('输入的数组大小不一致')

    return tuple(x[shuffle_indices] for x in args)


def split_batch(*args, **kwargs):
    batch_size = kwargs['batch_size']
    # 测试时只需要用户
    if len(args) == 1:
        tensor = args[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    # 训练时需要正例和负例
    else:
        for i in range(0, len(args[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in args)


# 推荐标准度量

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


# precision和recall度量
def RecallPrecision_ATk(ground, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(ground[i]) for i in range(len(ground))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


# NDCG度量
def NDCGatK_r(test_data, r, k):
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
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X):
    K_num = [20]
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in K_num:
        res = RecallPrecision_ATk(groundTrue, r, k + 1)
        pre.append(res['precision'])
        recall.append(res['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def test_performance(dataset, rec_model, b_size, topk):
    results = {'precision': np.zeros(len(topk)),
               'recall': np.zeros(len(topk)),
               'ndcg': np.zeros(len(topk))}
    user_data = dataset.test_data
    rec_model.eval()
    cores_num = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores_num)
    with torch.no_grad():
        users_id = list(user_data.keys())
        user_lis = []
        rating_lis = []
        ground_truth = []
        total_batch = len(users_id) // b_size + 1
        for batch_users in tqdm(split_batch(users_id, batch_size=b_size)):
            all_pos = dataset.getUserPois(batch_users)
            groundTrue = [user_data[u] for u in batch_users]
            batch_users_tensor = torch.Tensor(batch_users).long().to(loader.device)

            u_rating = rec_model.test_rating(batch_users_tensor)
            _, rating_k = torch.topk(u_rating, k=max(topk))

            del u_rating
            user_lis.append(batch_users)
            rating_lis.append(rating_k.cpu())
            ground_truth.append(groundTrue)
        assert total_batch == len(user_lis)
        X = zip(rating_lis, ground_truth)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users_id))
        results['precision'] /= float(len(users_id))
        results['ndcg'] /= float(len(users_id))

        print(results)
        return results


# 开始训练
epochs = 50
batch_size = 256
learning_rate = 0.001
layer_num = 3
latent_dim = 32
regular1 = 0.2
regular2 = 1e-4
max_K = [20]
GPR_net = GPR(loader, latent_dim, layer_num)
GPR_net.to(loader.device)
for p in GPR_net.parameters():
    if p.dim() > 1:
        #         nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)
# for name, param in GPR_net.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# 是否需要加入模型的xavier参数初始化
optimizer = optim.Adam(GPR_net.parameters(), lr=learning_rate)
for epoch in range(epochs):
    s = get_pos_neg_data(loader)
    user = torch.tensor(s[:, 0]).long().to(loader.device)
    pos = torch.tensor(s[:, 1]).long().to(loader.device)
    neg = torch.tensor(s[:, 2]).long().to(loader.device)
    loss_ = 0
    #     print(user.device)
    shuffle_user, shuffle_pos, shuffle_neg = shuffle_data(user, pos, neg)
    print('start_test')
    per_result = test_performance(loader, GPR_net, batch_size, max_K)
    print('start train epoch:', epoch)
    for batch_index, (batch_u, batch_pos, batch_neg) in enumerate(
            split_batch(shuffle_user, shuffle_pos, shuffle_neg, batch_size=batch_size)):
        optimizer.zero_grad()

        u_scores, batch_mse_loss, batch_bpr_loss, batch_bpr_reg_loss = GPR_net(batch_u, batch_pos, batch_neg)
        total_loss = batch_mse_loss + regular1 * batch_bpr_reg_loss + regular2 * batch_bpr_loss
        loss_ += total_loss
        total_loss.backward()
        optimizer.step()
    # if epoch%5==0:

    print(epoch, 'epoch loss:', loss_)