from __future__ import print_function

import copy

import torch.nn as nn
import torch.nn.functional as F
import math
# A number of functions/classes are adopted from: https://github.com/jingraham/neurips19-graph-protein-design
from torch.autograd import Variable
import torch
import numpy as np
from data import utils as du
from typing import Callable, List, Optional, Tuple
from chroma.layers.structure.protein_graph import ProteinGraph
from chroma.layers.structure import geometry,transforms
from models.ipa_pytorch import LightInvariantPointAttention,SInvariantPointAttention
from models import ipa_pytorch


from torch_scatter import scatter_mean
def u2(nodes,neighbor_features,neighbor_idx):
    # 假设 nodes, neighbor_features, neighbor_idx 已经定义
    # nodes 的形状为 [B, N, C], neighbor_features 的形状为 [B, N, K, C], neighbor_idx 的形状为 [B, N, K]

    B, N, K, C = neighbor_features.shape

    # 展平 neighbor_idx 和 neighbor_features
    # 对于 neighbor_idx，我们需要考虑批次中每个样本的偏移量
    flat_neighbor_idx = neighbor_idx + (torch.arange(B, device=neighbor_idx.device).view(B, 1, 1) * N)
    flat_neighbor_idx = flat_neighbor_idx.view(-1)

    flat_neighbor_features = neighbor_features.view(B * N * K, C)

    # 创建一个与 nodes 形状相同的张量来累加更新的特征
    updated_nodes = torch.zeros_like(nodes.view(B * N, C))

    # 使用 index_add_ 将邻居特征加到相应的节点上
    # 第一个参数为1表示按列加，flat_neighbor_idx 指定在哪些行上加，flat_neighbor_features 是加的值
    updated_nodes.index_add_(0, flat_neighbor_idx.view(-1), flat_neighbor_features)

    # 对于每个节点，计算它作为邻居被引用的次数，用于之后的平均计算
    counts = torch.zeros(B * N, device=nodes.device)
    counts.index_add_(0, flat_neighbor_idx.view(-1), torch.ones_like(flat_neighbor_idx, dtype=torch.float))

    # 避免除以零
    counts[counts == 0] = 1

    # 将累加的特征取平均
    updated_nodes /= counts.unsqueeze(-1)

    # 将 updated_nodes 重新塑形为原始 nodes 的形状
    updated_nodes = updated_nodes.view(B, N, C)

    return updated_nodes


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def chain_map_to_mask(C: torch.LongTensor) -> torch.Tensor:
    """Convert chain map into a mask.

    Args:
        C (torch.LongTensor): Chain map with shape
            `(num_batch, num_residues)`.

    Returns:
        mask (Tensor, optional): Mask tensor with shape
            `(num_batch, num_residues)`.
    """
    return (C > 0).type(torch.float32)



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def _S_to_seq(S, mask):
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)




def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=15, augment_eps=0.,  ):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        rbf_node_in, edge_ins = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embeddings = nn.Linear(edge_ins, edge_features, bias=False)


        self.norm_edges = nn.LayerNorm(edge_features)


    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx



    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF



    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        neg_dis = torch.exp(-D_A_B_neighbors)
        return RBF_A_B  # ,neg_dis

    def forward(self,  X, mask, residue_idx ):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]


        D_neighbors, E_idx = self._dist(Ca, mask)
        RBF_all = []

        RBF_all.append(self._get_rbf(Ca, Ca, E_idx))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O



        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]
        E_positional = self.embeddings(offset.long())

        # h_ve=gather_nodes(h_v, E_idx)

        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embeddings(E)
        E = self.norm_edges(E)



        return E


class EncoderLayer(nn.Module):
    def __init__(self, ipa,num_hidden, num_in, dropout=0.1,precision='16-mixed', scale=36,mode='notbinder'):
        super(EncoderLayer, self).__init__()
        self.precision=precision

        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)


        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        # sipa=ipa
        # sipa.c_s=ipadim

        # for not binder
        # ipadim=ipa.c_s
        # for binder mode
        if mode=='binder':
            ipadim=num_hidden
            ipa.c_s = ipadim
        else:
            ipadim=num_hidden
            ipa.c_s = ipadim
        self.lightipa_neigh = SInvariantPointAttention(ipa)
        self.neigh_ipa_ln = nn.LayerNorm(ipadim)  # neigh_ipa_ln
        self.neigh_transition = ipa_pytorch.StructureModuleTransition(c=ipadim

        )
        self.neigh_bb_update = ipa_pytorch.BackboneUpdate(ipadim, use_rot_updates=True)  # neigh_bb_update


        print('gnn hidden dim:',num_hidden)

        print('ipa dim:', ipadim)

    def forward(self, h_V, h_E, E_idx, mask_V, mask_attend,R,pos,kwargs):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))



        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            mask_V = mask_V.squeeze(-1)

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))


        # h_V_update,R=self.node_update(node_h=h_V,R=R,mask_i=mask_V,pos=pos,kwargs=kwargs)
        # h_V=h_V+h_V_update

        # fix fixed area
        h_v_update,R=self.neigh_update(h_V,edge_h=h_E,edge_idx=E_idx,R=R,mask_i=mask_V,mask_ij=mask_attend,fixed_mask=kwargs['fixed_mask'])
        h_V=h_V+h_v_update

        return h_V, h_E,R

    def node_update(self,node_h,R,mask_i,pos,kwargs):

        r_i=R.get_rots().get_rot_mats()
        t_i=R.get_trans()
        # node update
        if self.precision == '16-mixed':
            r_i=r_i.half()
            t_i=t_i.half()

        ipa_embed=self.lightipa(single_repr=node_h,
                                pairwise_repr=None ,
                                rotations=r_i, #.half()
                                translations=t_i,  #.half()
                                position_ids=pos,
                                mask=mask_i.bool())


        # ipa_embed=self.nodeffn(node_h)
        ipa_embed *= mask_i[..., None]
        node_embed = self.ipa_ln(node_h + ipa_embed) #+ ipa_embed
        node_embed = self.node_transition(node_embed)
        node_embed = node_embed * mask_i[..., None]


        if kwargs['fixed_mask'] is not None:
            update_mask =(1- kwargs['fixed_mask'][..., None])*mask_i[..., None]  # mark the pos which needed to be update
        else:
            update_mask = mask_i[..., None]

        node_rigid_update = self.bb_update(
            node_embed* update_mask)  # B,N 6
        R = R.compose_q_update_vec(
                node_rigid_update,update_mask)

        return node_embed,R


    def neigh_update(self,nodes,edge_h,edge_idx,R,mask_i,mask_ij,fixed_mask,fixed_area=True):
        '''
        for binder design
        fixed_area=True  , the fiexed area is not allowed to update

        for motif, the fiexed area is allowed to update

        '''
        #
        if fixed_mask==None:
            fixed_area=False

        r_i=R.get_rots().get_rot_mats()
        t_i=R.get_trans()



        b,n,k=mask_ij.shape
        mask_j=mask_ij.reshape(b*n,k) # B N K -> B*N K
        neigh_embed=edge_h.reshape(b*n,k,-1) # B N K num_hidden + num_in -> B*N K num_hidden + num_in

        R_j_init, t_j_init = transforms.collect_neighbor_transforms(
            r_i,  #.half()
            t_i, #.half()
            edge_idx
        )
        R_j_init=R_j_init.reshape(b*n,k,3,3)
        t_j_init = t_j_init.reshape(b * n, k,  3)
        pos_idx=edge_idx.reshape(b*n,k)
        RJ=du.create_rigid(R_j_init,t_j_init)
        neigh_ipa_embed,_ = self.lightipa_neigh(s=neigh_embed,
                                  z=None,
                                  r=RJ, #.half()
                                  position_ids=pos_idx,
                                  mask=mask_j)  # B*N K C


        neigh_ipa_embed =neigh_ipa_embed* mask_j[..., None]
        neigh_ipa_embedx=neigh_ipa_embed.reshape(b,n,k,-1)
        neigh_ipa_embed=u2(nodes,neigh_ipa_embedx,edge_idx)

        neigh_embed = self.neigh_ipa_ln( neigh_ipa_embed)
        neigh_embed = self.neigh_transition(neigh_embed)
        neigh_embed = neigh_embed #* mask_i[..., None]

        # print(mask_i.shape)
        if fixed_area:
            update_mask =mask_i[..., None] *(1- fixed_mask[..., None])# # mark the pos which needed to be update
        else:
            update_mask = mask_i[..., None]
        neigh_bb_update = self.neigh_bb_update(
            neigh_embed * update_mask)  # B*N  6


        R = R.compose_q_update_vec(
            neigh_bb_update, update_mask)


        return neigh_embed,R



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, precision='16-mixed', scale=36):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V



class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = nn.Linear(self.c_hidden, self.c_hidden, )
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden, )

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial
class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_in, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]

        s = self.relu(s)
        s = self.linear_in(s)


        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 1]
        s = self.linear_out(s)

        # [*, no_angles, 1]


        return s

class Pred_angle_Points(nn.Module):
    def __init__(self,c_s,c_resnet,no_resnet_blocks=2,no_angles=4,trans_scale_factor=10,a_epsilon=1e-6,**kwargs
                ):
        super(Pred_angle_Points, self).__init__()
        self.c_s=c_s
        self.c_resnet=c_resnet
        self.no_resnet_blocks=no_resnet_blocks
        self.no_angles=no_angles
        self.epsilon=a_epsilon

        self.trans_scale_factor=trans_scale_factor
        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(self, s,):  #


        # [*, N, 7, 2]
        angles = self.angle_resnet(s)


        return  angles


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class MLP(nn.Module):
    """Multilayer perceptron with variable input, hidden, and output dims.

    Args:
        dim_in (int): Feature dimension of input tensor.
        dim_hidden (int or None): Feature dimension of intermediate layers.
            Defaults to matching output dimension.
        dim_out (int or None): Feature dimension of output tensor.
            Defaults to matching input dimension.
        num_layers_hidden (int): Number of hidden MLP layers.
        activation (str): MLP nonlinearity.
            `'relu'`: Rectified linear unit.
            `'softplus'`: Softplus.
        dropout (float): Dropout rate. Default is 0.

    Inputs:
        h (torch.Tensor): Input tensor with shape `(..., dim_in)`

    Outputs:
        h (torch.Tensor): Input tensor with shape `(..., dim_in)`
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: Optional[int] = None,
        dim_out: Optional[int] = None,
        num_layers_hidden: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super(MLP, self).__init__()

        # Default is dimension preserving
        dim_out = dim_out if dim_out is not None else dim_in
        dim_hidden = dim_hidden if dim_hidden is not None else dim_out

        nonlinearites = {"relu": nn.ReLU, "softplus": nn.Softplus}
        activation_func = nonlinearites[activation]

        if num_layers_hidden == 0:
            layers = [nn.Linear(dim_in, dim_out)]
        else:
            layers = []
            for i in range(num_layers_hidden):
                d_1 = dim_in if i == 0 else dim_hidden
                layers = layers + [
                    nn.Linear(d_1, dim_hidden),
                    activation_func(),
                    nn.Dropout(dropout),
                ]
            layers = layers + [nn.Linear(dim_hidden, dim_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers(h)


class Graph(nn.Module):
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_encoder_layers=6,
                 k_neighbors=36,  dropout=0.1, precision='16-mixed', ipa=None, CSB=True,mode='base',**kwargs):
        super(Graph, self).__init__()
        '''
        forward for binder,CSB is True
        forward_fixed topo for base model, which use node update ,rcsb_cluster30_GNNfromIPAscratch_node_mixed_again
        forward_ss for motif, neigh update, rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto
        
        '''

        # Hyperparameters
        self.k_neighbors = k_neighbors

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim


        self.W_h = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        self.graph_builder = ProteinGraph(num_neighbors=k_neighbors)


        # Encoder layers
        if mode=='base_old':
            self.encoder_layers = nn.ModuleList([
                EncoderLayer_NODE(ipa,hidden_dim, hidden_dim * 2,dropout=dropout,precision=precision)
                for _ in range(num_encoder_layers)
            ])
        else:

            self.encoder_layers = nn.ModuleList([
                EncoderLayer(ipa,hidden_dim, hidden_dim * 2, dropout=dropout,precision=precision)
                for _ in range(num_encoder_layers)
            ])

        self.CSB=CSB
        if self.CSB:
            self.bfactor_pre=MLP( dim_in=hidden_dim,
                dim_hidden=node_features,
                dim_out=4, #for backbone 4 atoms
                num_layers_hidden=2,)

            self.aatype_pre=MLP( dim_in=hidden_dim,
                dim_hidden=node_features,
                dim_out=21, #for backbone 4 atoms
                num_layers_hidden=2,)


            self.chi_pred=Pred_angle_Points(c_s=hidden_dim,c_resnet=hidden_dim,)

        self.recycle=1



    def forward_fix_tpology(self,  C, R,node_h,edge_h,pos,mask,feature_layers, **kwargs):
        """ Graph-conditioned sequence model """

        mask_i = chain_map_to_mask(C)
        Ca=R.get_trans()*10 ## nm to ang
        E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
        edge_knn_h=gather_edges(edge_h,E_idx)


        h_V = self.W_h(node_h)
        h_E = self.W_e(edge_knn_h)

        # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        mask_attend=mask_attend*mask_ij



        for layer in self.encoder_layers:
            h_V, h_E,R = layer(h_V, h_E, E_idx, mask, mask_attend,R,pos,kwargs)


        return R#F.gumbel_softmax(logtis, tau=1, hard=False)   [logtis]


    def forward_fix_tpology_forbinder(self,  C, R,node_h,edge_h,pos,mask,recycle,feature_layers, frame_builder, rigids_nm_to_ang,**kwargs):
        """ Graph-conditioned  model
            we try use this finetune model to binder

        """

        mask_i = chain_map_to_mask(C)
        Ca=R.get_trans()*10 ## nm to ang
        E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
        edge_knn_h=gather_edges(edge_h,E_idx)


        h_V = self.W_h(node_h)
        h_E = self.W_e(edge_knn_h)

        # # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        mask_attend=mask_attend*mask_ij



        for layer in self.encoder_layers:
            h_V, h_E,R = layer(h_V, h_E, E_idx, mask, mask_attend,R,pos,kwargs)


        if self.CSB:
            # Decode b factor
            b_f=self.bfactor_pre(h_V)
            chi=self.chi_pred(h_V)
            S=self.aatype_pre(h_V)

            return R,b_f,chi,S


        else:
            return R,None,None,None,None,None

    def forward(self, C, R, node_h, edge_h, pos, mask, recycle, feature_layers,frame_builder, rigids_nm_to_ang, **kwargs):

        """ Graph-conditioned sequence/str model for binder
            graph will be calculated evary encoder layers
        """

        mask_i = chain_map_to_mask(C)
        input_feats = kwargs
        for recycle in range(self.recycle):
            #  rigids in ang
            curr_rigids = rigids_nm_to_ang(R)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()
            input_feats['trans_t'] = pred_trans
            input_feats['rotmats_t'] = pred_rotmats
            input_feats['trans_sc'] = pred_trans
            # feature
            X_t = frame_builder(pred_rotmats, pred_trans, C).float()
            # curr_rigids, node_embed, edge_embed, pos, node_mask, edge_mask, chain_idx = feature_layers(
            #     input_feats)  # curr_rigids is nm
            node_embed, edge_embed, edge_idx, mask_i, mask_ij = feature_layers(X_t, C, )

            # feature
            edge_h = edge_h + edge_embed
            node_h = node_h + node_embed
            h_V = self.W_h(node_h)
            for layer in self.encoder_layers:
                # to ang

                Ca = R.get_trans()
                E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
                edge_knn_h = gather_edges(edge_h, E_idx)

                h_E = self.W_e(edge_knn_h)

                # # Encoder is unmasked self-attention
                mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
                mask_attend = mask.unsqueeze(-1) * mask_attend
                mask_attend = mask_attend * mask_ij

                h_V, h_E, R = layer(h_V, h_E, E_idx, mask, mask_attend, R, pos, kwargs)

        if self.CSB:
            # Decode b factor
            b_f=self.bfactor_pre(h_V)
            chi=self.chi_pred(h_V)
            S=self.aatype_pre(h_V)



            return R,b_f,chi,S

        else:
            return R,None,None,None,None,None



    def forward_csb_noitr(self, C, R, node_h, edge_h, pos, mask, recycle, feature_layers,frame_builder, rigids_nm_to_ang, **kwargs):

        """ Graph-conditioned sequence model for binder
            graph will be calculated evary encoder layers
        """

        mask_i = chain_map_to_mask(C)
        input_feats = kwargs
        for recycle in range(self.recycle):
            #  rigids in ang
            curr_rigids = rigids_nm_to_ang(R)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()
            input_feats['trans_t'] = pred_trans
            input_feats['rotmats_t'] = pred_rotmats
            input_feats['trans_sc'] = pred_trans
            # feature
            X_t = frame_builder(pred_rotmats, pred_trans, C).float()
            # curr_rigids, node_embed, edge_embed, pos, node_mask, edge_mask, chain_idx = feature_layers(
            #     input_feats)  # curr_rigids is nm
            node_embed, edge_embed, edge_idx, mask_i, mask_ij = feature_layers(X_t, C, )

            # feature
            edge_h = edge_h + edge_embed
            node_h = node_h + node_embed
            h_V = self.W_h(node_h)

            Ca = R.get_trans()
            E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
            edge_knn_h = gather_edges(edge_h, E_idx)

            h_E = self.W_e(edge_knn_h)

            # # Encoder is unmasked self-attention
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            mask_attend = mask_attend * mask_ij


            for layer in self.encoder_layers:
                # to ang

                h_V, h_E, R = layer(h_V, h_E, E_idx, mask, mask_attend, R, pos, kwargs)

        if self.CSB:
            # Decode b factor
            b_f=self.bfactor_pre(h_V)
            chi=self.chi_pred(h_V)
            S=self.aatype_pre(h_V)


            return R,b_f,chi,S

        else:
            return R,None,None,None

    def forwardSS(self,  C, R,node_h,edge_h,pos,mask,recycle,feature_layers,rigids_nm_to_ang, **kwargs):
        """ Graph-conditioned sequence model for motif"""
        h_V=None
        mask_i = chain_map_to_mask(C)
        input_feats=kwargs
        for recycle in range(recycle):
            #  rigids in ang
            curr_rigids = rigids_nm_to_ang(R)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()
            input_feats['trans_t']=pred_trans
            input_feats['rotmats_t']=pred_rotmats
            input_feats['trans_sc']=pred_trans
            # feature
            curr_rigids, node_embed, edge_embed, pos, node_mask, edge_mask, chain_idx = feature_layers(input_feats)  #curr_rigids is nm


            # feature
            edge_h=edge_h+edge_embed
            node_h=node_h+node_embed
            if h_V is None:
                h_V = self.W_h(node_h)
            else:
                h_V = h_V + self.W_h(node_h)

            #he
            Ca = pred_trans
            E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
            edge_knn_h = gather_edges(edge_h, E_idx)
            h_E = self.W_e(edge_knn_h)

            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            mask_attend = mask_attend * mask_ij

            for layer in self.encoder_layers:
                # # Encoder is unmasked self-attention
                h_V, h_E,R = layer(h_V, h_E, E_idx, mask, mask_attend,R,pos,kwargs)


        return R#F.gumbel_softmax(logtis, tau=1, hard=False)   [logtis]


    def _positional_embeddings(self, pos,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        pse = []
        for idx in pos:
            d = idx

            frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=pos.device)
                * -(np.log(10000.0) / num_embeddings)
            )
            angles = d.unsqueeze(-1) * frequency
            E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
            pse.append(E)
        pse = torch.stack(pse)
        return pse


class Sidechain(nn.Module):
    def __init__(self, node_features, edge_features,
                 hidden_dim, num_encoder_layers=4,
                 k_neighbors=36,  dropout=0.1, precision='16-mixed', ipa=None,**kwargs):
        super(Sidechain, self).__init__()

        # Hyperparameters
        self.k_neighbors = k_neighbors

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        #features
        self.features=ProteinFeatures(edge_features, node_features,top_k=k_neighbors)

        self.W_h = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        self.graph_builder = ProteinGraph(num_neighbors=k_neighbors)


        # Encoder layers
        self.num_encoder_layers=num_encoder_layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(ipa,hidden_dim, hidden_dim * 2,dropout=dropout,precision=precision)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim ,dropout=dropout,precision=precision)
            for _ in range(num_encoder_layers)
        ])


        self.CSB=True
        if self.CSB:

            self.aatype_pre=MLP( dim_in=hidden_dim,
                dim_hidden=node_features,
                dim_out=21, #for backbone 4 atoms
                num_layers_hidden=2,)


            self.chi_pred=Pred_angle_Points(c_s=hidden_dim,c_resnet=hidden_dim)

        self.recycle=1




    def forward(self, C, R, node_embed,edge_embed,  pos, mask,feature_layers,frame_builder, rigids_nm_to_ang, **kwargs):
        '''
        C: (N, 3)
        R: (N, 3, 3)
        node_h: (N, node_features) from str generate stage

        '''

        """ Graph-conditioned sequence model for binder
            graph will be calculated evary encoder layers
        """

        mask_i = chain_map_to_mask(C)
        input_feats = kwargs
        for recycle in range(self.recycle):
            #  rigids in ang
            curr_rigids = rigids_nm_to_ang(R)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()
            input_feats['trans_t'] = pred_trans
            input_feats['rotmats_t'] = pred_rotmats
            input_feats['trans_sc'] = pred_trans
            # feature
            X_t = frame_builder(pred_rotmats, pred_trans, C).float()
            # curr_rigids, node_embed, edge_embed, pos, node_mask, edge_mask, chain_idx = feature_layers(
            #     input_feats)  # curr_rigids is nm
            #node_embed, edge_embed, edge_idx, mask_i, mask_ij = feature_layers(X_t, C, )

            # feature
            edge_h =  edge_embed
            # node_h =  node_embed
            h_V = self.W_h(node_embed)

            cl=0

            # to ang

            Ca = R.get_trans()
            E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
            edge_knn_h = gather_edges(edge_h, E_idx)

            # Get edge atoms features
            edge_knn_h = edge_knn_h + self.features( X_t, mask, kwargs['res_idx'])

            h_E = self.W_e(edge_knn_h)

            # # Encoder is unmasked self-attention
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            mask_attend = mask_attend * mask_ij



            for layer in self.encoder_layers:
                h_V, h_E, R = layer(h_V, h_E, E_idx, mask, mask_attend, R, pos, kwargs)


                # restore to ang and cal Xt
                if cl <self.num_encoder_layers-1:
                    # curr_rigids = rigids_nm_to_ang(R)
                    # pred_trans = curr_rigids.get_trans()
                    # pred_rotmats = curr_rigids.get_rots().get_rot_mats()
                    # X_t = frame_builder(pred_rotmats, pred_trans, C).float()

                    cl=cl+1

            for layer in self.decoder_layers:
                h_V = layer(h_V, h_E,  mask, mask_attend)





        if self.CSB:
            # Decode b factor

            chi=self.chi_pred(h_V)
            S=self.aatype_pre(h_V)


            return R,chi,S

        else:
            return R,None,None,None


    def forward_fixed(self, C, R, node_embed,edge_embed,  pos, mask,feature_layers,frame_builder, rigids_nm_to_ang, **kwargs):
        '''
        C: (N, 3)
        R: (N, 3, 3)
        node_h: (N, node_features) from str generate stage

        '''

        """ Graph-conditioned sequence model for binder
            graph will be calculated evary encoder layers
        """

        mask_i = chain_map_to_mask(C)
        input_feats = kwargs
        for recycle in range(self.recycle):
            #  rigids in ang
            curr_rigids = rigids_nm_to_ang(R)
            pred_trans = curr_rigids.get_trans()
            pred_rotmats = curr_rigids.get_rots().get_rot_mats()
            input_feats['trans_t'] = pred_trans
            input_feats['rotmats_t'] = pred_rotmats
            input_feats['trans_sc'] = pred_trans
            # feature
            X_t = input_feats['bbatoms_t']
            # curr_rigids, node_embed, edge_embed, pos, node_mask, edge_mask, chain_idx = feature_layers(
            #     input_feats)  # curr_rigids is nm
            #node_embed, edge_embed, edge_idx, mask_i, mask_ij = feature_layers(X_t, C, )

            # feature
            edge_h =  edge_embed
            # node_h =  node_embed
            h_V = self.W_h(node_embed)


            # to ang
            Ca = pred_trans
            E_idx, mask_ij = self.graph_builder(Ca.unsqueeze(-2), C)
            edge_knn_h = gather_edges(edge_h, E_idx)

            # Get edge atoms features
            edge_knn_h = edge_knn_h + self.features( X_t, mask, kwargs['res_idx'])

            h_E = self.W_e(edge_knn_h)

            # # Encoder is unmasked self-attention
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            mask_attend = mask_attend * mask_ij


            for layer in self.encoder_layers:


                h_V, h_E, _ = layer(h_V, h_E, E_idx, mask, mask_attend, R, pos, kwargs)



            for layer in self.decoder_layers:
                h_V = layer(h_V, h_E,  mask, mask_attend)





        if self.CSB:
            # Decode b factor

            chi=self.chi_pred(h_V)
            S=self.aatype_pre(h_V)


            return R,chi,S

        else:
            return R,None,None,None

    def _positional_embeddings(self, pos,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings

        pse = []
        for idx in pos:
            d = idx

            frequency = torch.exp(
                torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=pos.device)
                * -(np.log(10000.0) / num_embeddings)
            )
            angles = d.unsqueeze(-1) * frequency
            E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
            pse.append(E)
        pse = torch.stack(pse)
        return pse









# if __name__ == '__main__':
#     index = 'XACDEFGHIKLMNPQRSTVWY'
#     seq = 'RPALPDQAEMRLVFIDGDADEWLAGIEAARLDAMALSIHRYIRE'
#     s = []
#     for i in seq:
#         s.append(index.index(i))
#     print(s)
#     print("===> testing polar ...")
#
#     # out = polar(data)
#     # print(out.shape)