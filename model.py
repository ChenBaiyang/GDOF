import itertools as its
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale
import psutil
from scipy.stats import skew, kurtosis, entropy, lognorm
# from scipy.spatial.distance import cdist
# from sklearn.metrics import mutual_info_score
torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def relation_matrix_torch2(vec1, e):
    dist_matrix = torch.cdist(vec1, vec1, p=1)
    if e < 1e-6:
        return (dist_matrix < 1e-6).float()
    ave_dist = dist_matrix.mean()
    radius = min(ave_dist * e, 0.999)
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > radius] = 0
    return relation_matrix.float()


def relation_matrix_torch3(vec1, e):
    m, n, _ = vec1.shape
    dist_matrix = torch.cdist(vec1, vec1, p=1)
    dist_matrix[dist_matrix > 1 + 1e-6] = 0
    dist_matrix[e < 1e-6] = (dist_matrix[e < 1e-6] > 1e-6).float()
    ave_dist = dist_matrix.mean(dim=(1, 2))
    radius = (ave_dist * e).clip(1e-6, 0.999)
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > radius.reshape(len(e), 1, 1)] = 0
    return relation_matrix.float()


def relation_matrix_torch4(vec1, lamb):
    n, m = vec1.shape
    assert vec1.ptp(axis=0).max() < 1 + 1e-6
    vec1 = torch.from_numpy(vec1)
    dist_matrix = torch.cdist(vec1, vec1, p=2) / np.sqrt(m)
    ave_dist = dist_matrix.mean()
    radius = min(ave_dist * lamb, 0.999)
    relation_matrix = 1 - dist_matrix
    relation_matrix[dist_matrix > radius] = 0
    return relation_matrix.float()


def relation_matrix_torch5(vec1, lamb, e):
    vec1 = torch.from_numpy(vec1)
    n, m = vec1.shape
    numericals = np.where(e > 1e-6)[0]
    len_numericals = len(numericals)
    if len_numericals == m:  ### In case that all attributes are numericals
        return relation_matrix_torch4(vec1, lamb)
    else:
        nominals = np.where(e < 1e-6)[0]
        x_nominals = vec1[:, nominals].T.reshape(m - len_numericals, n, 1)
        dist_nominals = torch.cdist(x_nominals, x_nominals, p=1)
        dist_nominals = (dist_nominals > 1e-6).sum(dim=0)
        if len_numericals > 0:  ### In case that attributes are mixed
            x_numericals = vec1[:, numericals].T.reshape(len_numericals, n, 1)
            dist_numericals = torch.cdist(x_numericals, x_numericals, p=1)
            dist_numericals = torch.square(dist_numericals).sum(dim=0)
            # dist_matrix = torch.sqrt(dist_numericals + dist_nominals)
            # ### 如果不进行归一化，则距离取值范围将达到m**0.5，远远超过1
            dist_matrix = torch.sqrt(dist_numericals + dist_nominals) / np.sqrt(m)
        else:  ### In case that all attributes are nominals
            dist_matrix = torch.sqrt(dist_nominals) / np.sqrt(m)

        ave_dist = dist_matrix.mean()
        ### combined attributes distance with the same radius to that of numericals
        radius = min(ave_dist * lamb, 0.999)
        relation_matrix = 1 - dist_matrix
        relation_matrix[dist_matrix > radius] = 0
        return relation_matrix.float()


class WLOD(object):
    def __init__(self, data, nominals):
        n, m = data.shape
        self.nominals = nominals
        self.data = torch.from_numpy(data).float().T.reshape(m, n, 1)
        self.make_dist_matrix()
        self.alpha = 0.5

    def make_dist_matrix(self):
        m, n, _ = self.data.shape
        self.dist_rel_mat = torch.cdist(self.data, self.data, p=1)
        # if self.nominals.sum() > 0:
        #     self.dist_rel_mat[self.nominals] = (self.dist_rel_mat[self.nominals] > 1e-6).float()

        for idx, i in enumerate(self.nominals):
            if i:
                self.dist_rel_mat[idx] = self.dist_rel_mat[idx] > 1e-6

        self.dist_rel_mat *= -1
        self.dist_rel_mat += 1


    def fit_lamb(self, rel_mat, pos=None, neg=None):
        """This is label-informed fuzzy radius.
        """
        # pos, neg = [309, 204, 207, 381, 420], [22, 47, 51, 119, 148, 263, 287, 314, 325, 431]
        pos_m = rel_mat[pos]
        neg_m = rel_mat[neg]
        pos_m_sorted = -np.sort(-pos_m, axis=None)
        neg_m_sorted = -np.sort(-neg_m, axis=None)
        pos_m_sorted[-1] = 0
        neg_m_sorted[-1] = 0
        pos_cum = np.cumsum(pos_m_sorted)
        neg_cum = np.cumsum(neg_m_sorted)

        # lambs = np.arange(1, 0, -0.01)
        lambs = np.arange(0.999, 0, -0.001)
        # lambs = np.arange(1, 0, -0.001)
        idx_pos = [np.where(pos_m_sorted < i)[0][0] for i in lambs + 1e-6]
        idx_neg = [np.where(neg_m_sorted < i)[0][0] for i in lambs + 1e-6]

        pos_lambs = pos_cum[idx_pos]
        neg_lambs = neg_cum[idx_neg]
        val_lambs = neg_lambs/len(neg_m_sorted)  - pos_lambs/len(pos_m_sorted)
        best_idx = np.argmax(val_lambs)
        lamb = round(1 - lambs[best_idx], 3)
        print(lamb)
        return lamb


    def Fuzzy_granule_density(self):
        m, n, _ = self.dist_rel_mat.shape
        self.FGD = torch.zeros((m, n))
        for i in range(m):
            neighbors = (self.dist_rel_mat[i] > 0).float()
            r_m_sum = self.dist_rel_mat[i].sum(dim=-1)
            self.FGD[i] = torch.square(r_m_sum) * neighbors.mean(dim=-1)
            self.FGD[i] /= torch.matmul(neighbors, r_m_sum)
        return self.FGD

    def fit(self, pos_idx, n_neg=100):
        m, n, _ = self.dist_rel_mat.shape
        neg_idx = np.setdiff1d(np.arange(n), pos_idx)
        prob = self.dist_rel_mat.mean((0,1))[neg_idx]
        prob = torch.nn.functional.softmax(prob, dim=0, dtype=torch.float64)
        np.random.seed(1)
        neg_idx = np.random.choice(neg_idx, n_neg, replace=False, p=prob)
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx

        for i in range(m):
            if self.nominals[i]:
                continue
            temp_rel_mat = self.dist_rel_mat[i]
            lamb_i = self.fit_lamb(temp_rel_mat, pos=pos_idx, neg=neg_idx)
            temp_rel_mat[temp_rel_mat < 1-lamb_i] = 0
            self.dist_rel_mat[i] = temp_rel_mat

        self.Fuzzy_granule_density()

        low_appr_union = np.zeros((m, n), dtype=np.float32)
        for l in range(m):
            M_R_B = self.dist_rel_mat[l]
            M_R_B_N = 1 - M_R_B
            low_appr_union[l, pos_idx] = torch.min(M_R_B_N[pos_idx][:, neg_idx], dim=1)[0]
            low_appr_union[l, neg_idx] = torch.min(M_R_B_N[neg_idx][:, pos_idx], dim=1)[0]
        # the upper appr intersection for the opposite class is 1-low_appr_union:
        upp_appr_inter = 1 - low_appr_union

        ### old policy for alpha
        # self.rho = (low_appr_union / (1 + upp_appr_inter)).mean(axis=1)  # elementwise calculation
        # self.rho[pos_idx] *= self.alpha

        # improved policy for alpha
        self.rho = (low_appr_union / (1 + upp_appr_inter))

    def predict_score(self, test_idx, alpha=0.5):
        pos_idx, neg_idx = self.pos_idx, self.neg_idx
        rho = self.rho.copy()
        rho[:, pos_idx] *= alpha
        rho[:, pos_idx] /= len(pos_idx)
        rho[:, neg_idx] *= (1 - alpha)
        rho[:, neg_idx] /= len(neg_idx)
        rho = rho.sum(axis=1)
        # print(rho.sum())

        od = 1 - (self.FGD.T * rho).mean(axis=1)
        return od[test_idx]


class GDOF(object):
    def __init__(self, data, nominals):
        n, m = data.shape
        self.nominals = nominals
        self.data = torch.from_numpy(data).float().T.reshape(m, n, 1)
        self.make_dist_matrix()

    def make_dist_matrix(self):
        m, n, _ = self.data.shape
        self.dist_rel_mat = torch.cdist(self.data, self.data, p=1)
        # if self.nominals.sum() > 0:
        #     self.dist_rel_mat[self.nominals] = (self.dist_rel_mat[self.nominals] > 1e-6).float()

        for idx, i in enumerate(self.nominals):
            if i:
                self.dist_rel_mat[idx] = self.dist_rel_mat[idx] > 1e-6

        self.dist_rel_mat *= -1
        self.dist_rel_mat += 1
        del self.data


    def fit_lamb2(self, dist_mat, pos=None, neg=None):
        pos_mat = dist_mat[pos]
        neg_mat = dist_mat[neg]
        pos_mat_sorted = -np.sort(-pos_mat, axis=None)
        neg_mat_sorted = -np.sort(-neg_mat, axis=None)
        pos_mat_sorted[-1] = 0
        neg_mat_sorted[-1] = 0
        pos_cum = np.cumsum(pos_mat_sorted)
        neg_cum = np.cumsum(neg_mat_sorted)

        # lambs = np.arange(1, 0, -0.01)
        lambs = np.arange(0.999, 0, -0.001)
        # lambs = np.arange(1, 0, -0.001)
        idx_pos = [np.where(pos_mat_sorted < i)[0][0] for i in lambs + 1e-6]
        idx_neg = [np.where(neg_mat_sorted < i)[0][0] for i in lambs + 1e-6]
        print(idx_pos)
        exit()

        pos_lambs = pos_cum[idx_pos]
        neg_lambs = neg_cum[idx_neg]
        val_lambs = neg_lambs/len(neg_mat_sorted)  - pos_lambs/len(pos_mat_sorted)
        best_idx = np.argmax(val_lambs)
        lamb = round(1 - lambs[best_idx], 3)
        print(lamb)
        return lamb

    def fit_lamb3(self, dist_mat, pos=None, neg=None):
        pos_mat = dist_mat[pos]
        neg_mat = dist_mat[neg]
        pos_mat_sorted = -np.sort(-pos_mat, axis=None)
        neg_mat_sorted = -np.sort(-neg_mat, axis=None)
        pos_mat_sorted[-1] = 0
        neg_mat_sorted[-1] = 0
        pos_cum = np.cumsum(pos_mat_sorted)
        neg_cum = np.cumsum(neg_mat_sorted)

        candidate_lambs = np.arange(0.999, 0, -0.001).round(4)
        len_candidate = len(candidate_lambs)
        idx_pos = np.zeros(len_candidate, dtype=np.int)
        idx_neg = np.zeros_like(idx_pos)
        lamb_idx = 0
        lamb_i = candidate_lambs[lamb_idx]
        for idx2, j in enumerate(pos_mat_sorted):
            if lamb_i > j:   # 这里逻辑错误，应该用while循环
                idx_pos[lamb_idx] = idx2-1
                lamb_idx += 1
                if lamb_idx == len_candidate:
                    break
                lamb_i = candidate_lambs[lamb_idx]

        lamb_idx = 0
        lamb_i = candidate_lambs[lamb_idx]
        for idx2, j in enumerate(neg_mat_sorted):
            if lamb_i > j:
                idx_neg[lamb_idx] = idx2-1
                lamb_idx += 1
                if lamb_idx == len_candidate:
                    break
                lamb_i = candidate_lambs[lamb_idx]

        pos_lambs = pos_cum[idx_pos]
        neg_lambs = neg_cum[idx_neg]
        val_lambs = neg_lambs/len(neg_mat_sorted)  - pos_lambs/len(pos_mat_sorted)
        best_idx = np.argmax(val_lambs)
        lamb = round(1 - candidate_lambs[best_idx], 3)
        print(lamb)
        return lamb

    def fit_lamb(self, pos_mat, neg_mat):
        pos_mat_sorted = -np.sort(-pos_mat, axis=None)
        neg_mat_sorted = -np.sort(-neg_mat, axis=None)
        pos_mat_sorted[-1] = 0
        neg_mat_sorted[-1] = 0
        pos_cum = np.cumsum(pos_mat_sorted)
        neg_cum = np.cumsum(neg_mat_sorted)

        candidate_lambs = np.arange(0.999, 0, -0.001).round(4)
        len_candidate = len(candidate_lambs)
        idx_pos = np.zeros(len_candidate, dtype=np.int)
        idx_neg = np.zeros_like(idx_pos)
        lamb_idx = 0
        lamb_i = candidate_lambs[lamb_idx]

        for idx, j in enumerate(pos_mat_sorted):
            while lamb_i > j:           #为了减少内层循环的运算量，不添加提前终止条件，直接遍历完整个 pos_mat_sorted列表
                if lamb_idx < len_candidate:
                    idx_pos[lamb_idx] = idx
                    lamb_idx += 1
                    if lamb_idx < len_candidate:
                        lamb_i = candidate_lambs[lamb_idx]
                else:
                    break

        lamb_idx = 0
        lamb_i = candidate_lambs[lamb_idx]
        for idx, j in enumerate(neg_mat_sorted):
            while lamb_i > j:
                if lamb_idx < len_candidate:
                    idx_neg[lamb_idx] = idx
                    lamb_idx += 1
                    if lamb_idx < len_candidate:
                        lamb_i = candidate_lambs[lamb_idx]
                else:
                    break

        pos_lambs = pos_cum[idx_pos-1]
        neg_lambs = neg_cum[idx_neg-1]
        val_lambs = neg_lambs/len(neg_mat_sorted)  - pos_lambs/len(pos_mat_sorted)
        best_idx = np.argmax(val_lambs)
        lamb = round(1 - candidate_lambs[best_idx], 3)
        # print(lamb)
        return lamb

    def fit_lambs2(self, pos=None, neg=None):
        numericals = np.logical_not(self.nominals)
        m = numericals.sum()
        pos_mat = self.dist_rel_mat[numericals][:,pos].reshape(m, -1)
        neg_mat = self.dist_rel_mat[numericals][:,neg].reshape(m, -1)
        pos_mat_sorted = -np.sort(-pos_mat, axis=1)
        neg_mat_sorted = -np.sort(-neg_mat, axis=1)
        pos_mat_sorted[:, -1] = 0
        neg_mat_sorted[:, -1] = 0
        pos_cum = np.cumsum(pos_mat_sorted, axis=1)
        neg_cum = np.cumsum(neg_mat_sorted, axis=1)

        ## 这种方法别比逐一计算还更慢……
        candidate_lambs = np.arange(0.999, 0, -0.001)
        idx_pos = np.array([np.argmin(np.maximum(pos_mat_sorted - i, 0), axis=1) for i in candidate_lambs])
        idx_neg = np.array([np.argmin(np.maximum(neg_mat_sorted - i, 0), axis=1) for i in candidate_lambs])

        lambs = [0] * m
        for i in range(m):
            pos_lambs = pos_cum[i][idx_pos[:,i]]
            neg_lambs = neg_cum[i][idx_neg[:,i]]
            val_lambs = neg_lambs/neg_mat_sorted.shape[1]  - pos_lambs/pos_mat_sorted.shape[1]
            best_idx = np.argmax(val_lambs)
            lamb_i = round(1 - candidate_lambs[best_idx], 3)
            lambs[i] = lamb_i
            # print(lamb)
        return lambs

    def fit_lambs(self, pos=None, neg=None):
        numericals = np.logical_not(self.nominals)
        m = numericals.sum()
        pos_mat = self.dist_rel_mat[numericals][:,pos].reshape(m, -1)
        neg_mat = self.dist_rel_mat[numericals][:,neg].reshape(m, -1)
        pos_mat_sorted = -np.sort(-pos_mat, axis=1)
        neg_mat_sorted = -np.sort(-neg_mat, axis=1)
        pos_mat_sorted[:, -1] = 0
        neg_mat_sorted[:, -1] = 0
        pos_cum = np.cumsum(pos_mat_sorted, axis=1)
        neg_cum = np.cumsum(neg_mat_sorted, axis=1)

        candidate_lambs = np.arange(0.999, 0, -0.001).round(4)
        len_candidate = len(candidate_lambs)
        idx_pos = np.zeros((len_candidate, m), dtype=np.int)
        idx_neg = np.zeros_like(idx_pos)
        for idx1, vals in enumerate(pos_mat_sorted):
            lamb_idx = 0
            lamb_i = candidate_lambs[lamb_idx]
            for idx, j in enumerate(vals):
                if lamb_i > j:              # 这里逻辑错误，应该用while循环
                    idx_pos[lamb_idx, idx1] = idx-1
                    lamb_idx += 1
                    if lamb_idx == len_candidate:
                        break
                    lamb_i = candidate_lambs[lamb_idx]

        for idx1, vals in enumerate(neg_mat_sorted):
            lamb_idx = 0
            lamb_i = candidate_lambs[lamb_idx]
            for idx, j in enumerate(vals):
                if lamb_i > j:
                    idx_neg[lamb_idx, idx1] = idx-1
                    lamb_idx += 1
                    if lamb_idx == len_candidate:
                        break
                    lamb_i = candidate_lambs[lamb_idx]

        lambs = [0] * m
        for i in range(m):
            pos_lambs = pos_cum[i][idx_pos[:,i]]
            neg_lambs = neg_cum[i][idx_neg[:,i]]
            val_lambs = neg_lambs/neg_mat_sorted.shape[1]  - pos_lambs/pos_mat_sorted.shape[1]
            best_idx = np.argmax(val_lambs)
            lamb_i = round(1 - candidate_lambs[best_idx], 3)
            lambs[i] = lamb_i
            print(lamb_i)
        return lambs


    def Fuzzy_granule_density(self):
        m, n, _ = self.dist_rel_mat.shape
        self.FGD = torch.zeros((m, n))
        for i in range(m):
            neighbors = (self.dist_rel_mat[i] > 0).float()
            r_m_sum = self.dist_rel_mat[i].sum(dim=-1)
            self.FGD[i] = torch.square(r_m_sum) * neighbors.mean(dim=-1)
            self.FGD[i] /= torch.matmul(neighbors, r_m_sum)
        return self.FGD


    def fit(self, pos_idx, n_neg=100):
        m, n, _ = self.dist_rel_mat.shape
        unlabeled = np.setdiff1d(np.arange(n), pos_idx)
        prob = self.dist_rel_mat.mean((0,1))[unlabeled]
        prob = torch.nn.functional.softmax(prob, dim=0, dtype=torch.float64)
        n_neg = min(n_neg, int(len(unlabeled) * 0.8))
        np.random.seed(1)
        neg_idx = np.random.choice(unlabeled, n_neg, replace=False, p=prob)
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx

        # for i in range(m):
        #     if self.nominals[i]:
        #         continue
        #     temp_dist_mat = self.dist_rel_mat[i]
        #     lamb_i = self.fit_lamb2(temp_dist_mat, pos=pos_idx, neg=neg_idx)
        #     temp_dist_mat[temp_dist_mat < 1-lamb_i] = 0
        #     self.dist_rel_mat[i] = temp_dist_mat

        # for i in range(m):
        #     if self.nominals[i]:
        #         continue
        #     temp_dist_mat = self.dist_rel_mat[i]
        #     pos_mat = temp_dist_mat[pos_idx]
        #     neg_mat = temp_dist_mat[neg_idx]
        #     lamb_i = self.fit_lamb(pos_mat, neg_mat)
        #     temp_dist_mat[temp_dist_mat < 1-lamb_i] = 0
        #     self.dist_rel_mat[i] = temp_dist_mat

        for i in range(m):
            if not self.nominals[i]:
                pos_mat = self.dist_rel_mat[i, pos_idx]
                neg_mat = self.dist_rel_mat[i, neg_idx]
                lamb_i = self.fit_lamb(pos_mat, neg_mat)
                indices = self.dist_rel_mat[i] < 1 - lamb_i
                self.dist_rel_mat[i, indices] = 0

        self.Fuzzy_granule_density()

    def fit2(self, pos_idx, n_neg=100):
        m, n, _ = self.dist_rel_mat.shape
        unlabeled = np.setdiff1d(np.arange(n), pos_idx)
        prob = self.dist_rel_mat.mean((0,1))[unlabeled]
        prob = torch.nn.functional.softmax(prob, dim=0, dtype=torch.float64)
        n_neg = min(n_neg, int(len(unlabeled) * 0.8))
        np.random.seed(1)
        neg_idx = np.random.choice(unlabeled, n_neg, replace=False, p=prob)
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx

        numericals = np.logical_not(self.nominals)
        numerical_idx = np.arange(m)[numericals]
        if numericals.sum() > 0:
            lambs = self.fit_lambs(pos=pos_idx, neg=neg_idx)
            for i in range(len(lambs)):
                idx = numerical_idx[i]
                temp_dist_mat = self.dist_rel_mat[idx]
                temp_dist_mat[temp_dist_mat < 1-lambs[i]] = 0
                self.dist_rel_mat[idx] = temp_dist_mat

        self.Fuzzy_granule_density()


    def predict_score(self, test_idx, alpha=0.5):
        pos_idx, neg_idx = self.pos_idx, self.neg_idx
        gamma = alpha * self.FGD.T[neg_idx].mean(0) - (1-alpha) * self.FGD.T[pos_idx].mean(0)
        od = 1 - (self.FGD.T * gamma).mean(axis=1)
        return od[test_idx]
