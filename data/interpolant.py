import torch
import os

import numpy as np
from tqdm import tqdm

from data import so3_utils
from data import utils as du

import copy
from analysis.mask import design_masks,create_binder_mask
from scipy.optimize import linear_sum_assignment
from chroma.data.protein import Protein
from models.symmetry import SymGen
from chroma.layers.structure.mvn import BackboneMVNResidueGas
from scipy.spatial.transform import Rotation as R
from chroma.layers.structure.backbone import FrameBuilder
from models.noise_schedule import OTNoiseSchedule
import torch.nn.functional as F
from data.average_rot import batch_masked_karcher_mean as parallel_karcher_mean
def maskwise_avg_replace(tensor, mask):
    # tensor: [B, N, ...]
    # mask: [B, N]
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    inverse_mask = (mask == 0)
    avg_val = tensor[inverse_mask.expand(tensor.shape)].mean(dim=0)
    return torch.where(inverse_mask, avg_val.view(*([1] * (tensor.dim() - avg_val.dim())), *avg_val.shape), tensor)


def round_to(x, base=50):
    """四舍五入到最接近的 base（默认 50）的倍数。"""
    return int(base * round(float(x) / base))

def steps_by_length(L):
    """
    根据序列长度 L（aa）返回建议步长（不考虑 schedule）：
      - L < 80               -> 100
      - 80 ≤ L < 100         -> 200
      - 100 ≤ L ≤ 200        -> 200 + 三角峰（峰高300，中心150，两端100/200为0）
      - 200 < L ≤ 250        -> 200
      - 250 < L < 300        -> 500
      - 300 ≤ L < 350        -> 从500线性过渡到1000
      - 350 ≤ L ≤ 400        -> 1000
      - L > 400              -> 1000（需要更稳可改成1500）
    返回值四舍五入到最接近的50步。
    """
    L = float(L)

    # 1) very short
    if L < 80:
        return 100
    if L < 100:
        return 200

    # 2) 100~200：200 底座 + 三角峰（峰高=300，中心=150，两端=100/200 为 0）
    if 100 <= L <= 200:
        c, A, left, right = 150.0, 300.0, 100.0, 200.0
        w = (right - left) / 2.0  # = 50
        peak = max(0.0, 1.0 - abs(L - c) / w) * A
        return round_to(200.0 + peak)

    # 3) 200~250：固定 200
    if L <= 250:
        return 200

    # 4) 250~300：固定 500
    if L < 300:
        return 500

    # 5) 300~350：从 500 线性过渡到 1000
    if L < 350:
        steps = 500.0 + (L - 300.0) * (1000.0 - 500.0) / (350.0 - 300.0)
        return round_to(steps)

    # 6) 350~400：1000（如需更稳可改成1500）
    if L <= 400:
        return 1000

    # 7) >400：默认 1000（按需调成 1500）
    return 1000


def karcher_mean_quaternion(quats, max_iter=50, tol=1e-6):
    def log_map(q, mu):
        dot = (q * mu).sum(-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        v = q - dot * mu
        v = F.normalize(v, dim=-1)
        return theta * v

    def exp_map(v, mu):
        theta = torch.norm(v, dim=-1, keepdim=True)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        v_norm = F.normalize(v, dim=-1)
        return cos_theta * mu + sin_theta * v_norm

    q_mean = quats[0].clone()
    for _ in range(max_iter):
        log_vs = log_map(quats, q_mean)  # [K, 4]
        v_mean = log_vs.mean(dim=0, keepdim=True)
        delta = v_mean.norm()
        if delta < tol:
            break
        q_mean = exp_map(v_mean, q_mean)
        q_mean = F.normalize(q_mean, dim=-1)
    return q_mean[0]

def batch_masked_karcher_mean(rot_mats: torch.Tensor, fixed_mask: torch.Tensor) -> torch.Tensor:
    """
    rot_mats: [B, N, 3, 3]
    fixed_mask: [B, N] -> 0 indicates positions to be averaged and replaced
    """
    B, N, _, _ = rot_mats.shape
    device = rot_mats.device
    rot_out = rot_mats.clone()

    for n in range(N):
        # 找出该位置 n，所有 batch 中 fixed_mask==0 的旋转
        valid_indices = (fixed_mask[:, n] == 0).nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            continue  # 没有需要替换的

        selected_rot = rot_mats[valid_indices, n]  # [K, 3, 3]
        # 转为 numpy 旋转矩阵
        rots_np = selected_rot.detach().cpu().numpy()
        quats_np = R.from_matrix(rots_np).as_quat()  # [K, 4]

        quats = torch.tensor(quats_np, dtype=torch.float32, device=device)
        # 做 Karcher mean
        mean_quat = karcher_mean_quaternion(quats)  # [4]
        # 转回旋转矩阵
        mean_rot = R.from_quat(mean_quat.cpu().numpy()).as_matrix()
        mean_rot = torch.tensor(mean_rot, dtype=torch.float32, device=device)  # [3, 3]

        # 替换所有 batch 中该位置的 rot_mats
        for b in valid_indices:
            rot_out[b, n] = mean_rot

    return rot_out


def SS_mask(ss, fixed_mask,design=False):
    if  design:
        return ss.clone() *fixed_mask
    else:
        th = 0.5
    # 计算mask的全局比例
    mask_percentage = th

    # 创建一个与fixed_mask形状相同的随机矩阵
    random_matrix = torch.rand_like(fixed_mask.float())

    # 计算需要被mask的阈值
    threshold = torch.quantile(random_matrix[fixed_mask == 0], mask_percentage)

    # 创建最终的mask，对fixed_mask为0且随机值小于阈值的位置进行mask
    final_mask = (random_matrix < threshold) & (fixed_mask == 0)

    # 将ss中对应final_mask为True的位置设置为0
    new_ss = ss.clone()  # 复制ss避免修改原始数据
    new_ss[final_mask] = 0

    return new_ss
def update_connected_regions_batch(tensor):
    # 初始化结果张量，与输入张量相同的形状
    B, N = tensor.shape
    result = torch.zeros_like(tensor)

    for b in range(B):  # 遍历每个批次
        region_count = 0  # 重置连通区域计数器
        prev_val = 0  # 重置上一个值
        for n in range(N):  # 遍历序列中的每个元素
            # 如果当前值为1且上一个值为0，那么我们遇到了一个新的连通区域
            if tensor[b, n] == 1 and prev_val == 0:
                region_count += 1
            # 更新结果张量的值
            result[b, n] = region_count if tensor[b, n] == 1 else 0
            # 更新上一个值
            prev_val = tensor[b, n]

    return result

def generate_batch_constrained_points_torch(batch_size, n_points, device,base_distance=1.27, constraint_factor=0.0028):
    # Initialize all points at the origin for all batches
    points = torch.zeros(batch_size, n_points, 3,device=device)

    # Generate a random direction for each point in each batch
    random_directions = torch.randn(batch_size, n_points, 3,device=device)
    norms = torch.norm(random_directions, dim=2, keepdim=True)
    random_directions /= norms

    # Apply constraints and calculate the points iteratively (due to dependency on previous point)
    for i in range(1, n_points):
        constraints = -points[:, i-1, :] * constraint_factor * (i ** -0.1)
        adjusted_directions = random_directions[:, i, :] + constraints
        adjusted_directions /= torch.norm(adjusted_directions, dim=1, keepdim=True)

        points[:, i, :] = points[:, i-1, :] + adjusted_directions * base_distance
    points=torch.tensor(points,device=device)
    points=points - torch.mean(points, dim=-2, keepdims=True)
    return points
def _centered_gaussian(num_batch, num_res, atoms,device='cuda'):
    if atoms is not None:
        noise = torch.randn(num_batch, num_res,atoms, 3, device=device)
        return noise - torch.mean(noise, dim=[1,2], keepdims=True)
    else:
        noise = torch.randn(num_batch, num_res, 3, device=device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        R.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


def save_pdb_chain(xyz,chain_idx, pdb_out="out.pdb"):


    chains='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY",chains[int(chain_idx[a]-1)],a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()
def save_pdb(xyz, pdb_out="out.pdb"):
    pdb_out=pdb_out
    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()




def invert_noise_via_optimization(model, batch_copy, num_opt_steps=500, lr=1e-3):
    """
    反解高斯噪声: 给定目标真实坐标，反向优化获得初始高斯噪声。
    只对fixed_mask=1的部分进行优化，保持原始batch结构。
    """
    model.eval()
    

    
    # 获取fixed_mask和设备
    fixed_mask = batch_copy['fixed_mask']
    device = fixed_mask.device
    
    # 获取真实目标坐标
    trans_true = batch_copy['trans_1']
    rotmats_true = batch_copy['rotmats_1']
    
    # 创建完整尺寸的噪声张量
    trans_noise = batch_copy['trans_t']
    rotmats_noise = batch_copy['rotmats_t']
    
    # 设置时间t=0
    t_0 = torch.zeros_like(batch_copy['t'] if 't' in batch_copy else torch.zeros((trans_true.shape[0], 1), device=device))
    batch_copy['t'] = t_0
    
    # 将噪声张量中fixed_mask=1的部分设为可训练参数
    # 为此我们创建一个噪声"容器"并通过.data来更新它
    trans_noise.requires_grad_(True)
    rotmats_noise.requires_grad_(True)
    
    # 只优化fixed_mask=1部分的参数
    optimizer = torch.optim.Adam([
        {'params': [trans_noise], 'lr': lr},
        {'params': [rotmats_noise], 'lr': lr}
    ])
    
    for step in tqdm(range(num_opt_steps), desc="Optimizing noise"):
        optimizer.zero_grad()
        
        # 更新batch中的噪声
        batch_copy['trans_t'] = trans_noise
        batch_copy['rotmats_t'] = rotmats_noise
        
        # 前向传播
        model_out = model(batch_copy, recycle=1, is_training=True)
        
        # 只计算fixed_mask=1部分的损失
        pred_trans = model_out['pred_trans']
        pred_rotmats = model_out['pred_rotmats']
        
        # 创建只包含fixed_mask=1部分的损失掩码
        loss_mask_trans = fixed_mask.unsqueeze(-1).expand_as(trans_true)
        loss_mask_rotmats = fixed_mask.unsqueeze(-1).unsqueeze(-1).expand_as(rotmats_true)
        
        # 计算有掩码的损失
        loss_trans = (((pred_trans - trans_true) ** 2) * loss_mask_trans.float()).sum() / loss_mask_trans.float().sum()
        loss_rot = (((pred_rotmats - rotmats_true) ** 2) * loss_mask_rotmats.float()).sum() / loss_mask_rotmats.float().sum()
        loss = loss_trans + loss_rot
        
        # 反向传播
        loss.backward()
        
        # 手动将梯度置零（对于fixed_mask=0的部分）
        with torch.no_grad():
            trans_noise.grad.masked_fill_(~loss_mask_trans, 0)
            rotmats_noise.grad.masked_fill_(~loss_mask_rotmats, 0)
        
        # 更新参数
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}, loss: {loss.item():.5f}, trans_loss: {loss_trans.item():.5f}, rot_loss: {loss_rot.item():.5f}")
    
    # 返回优化后的噪声
    return trans_noise.detach(), rotmats_noise.detach()

class Interpolant_10:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

        self.backbone_init = BackboneMVNResidueGas().cuda()
        # Frame trunk
        self.frame_builder = FrameBuilder()

        # Noise schedule
        self.noise_perturb=OTNoiseSchedule()



        self._eps=1e-6
    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_bbatoms(self, trans_1, chain_idx, t, res_mask):
        """
        corrupt backbone atoms with a batchot, corrupt them Ca, and others respectively
        so other backbone atoms is not so far from Ca, to elucate the influence of searching space
        Args:trans_1:data  [B,N,4,3],trans_0:0

        Notes: 1-t used for gt, which is same direction with my paper


        """
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # 1. sample Ca and use batchot to solve it to get trans_t
        trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
        trans_0 = trans_nm_0 * rg
        trans_1_ca = trans_1[..., 1, :]
        trans_0 = self._batch_ot(trans_0, trans_1_ca, res_mask)
        # trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1[..., 1, :]

        trans_z = trans_0 / rg  # nm _SCALE

        # 2. sample N C O and use batchot to solve it to get others_t
        num_batch = chain_idx.shape[0]
        num_residues = chain_idx.shape[1]

        # noisy_rotmats = self.igso3.sample(
        #     torch.tensor([1.5]),
        #     num_batch * num_residues
        # ).to(self._device)
        # noisy_rotmats = noisy_rotmats.reshape(num_batch, num_residues, 3, 3)

        #so3z=self.frame_builder(noisy_rotmats, torch.zeros_like(trans_0), chain_idx)


        z = torch.rand(num_batch, num_residues, 4, 3).to(trans_1.device)
        mask = torch.zeros(num_batch, num_residues, 4, ).to(trans_1.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE

        # others_1 = trans_1 - trans_1[..., 1, :].unsqueeze(-2).repeat(1, 1, 4, 1)  #ANG_SCALE
        # others_1 = others_1/ du.NM_TO_ANG_SCALE  # nm_SCALE
        # others_t = (1 - t[..., None]) * others_z + t[..., None] * others_1  # nm_SCALE

        # 3. combine and transform to resgas
        self.backbone_init._register(stddev_CA=rg, device=self._device)
        z = mask[..., None] * trans_z.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bbatoms = self.backbone_init.sample(chain_idx, Z=z)
        xt = (1 - t[..., None, None]) *bbatoms   + t[..., None, None] *  trans_1  # nm_SCALE

        # forward in z space
        # z_0=self.backbone_init._multiply_R_inverse(trans_1,chain_idx)
        # z_hat=(1 - t[..., None, None]) * z_0 + t[..., None, None] * z  # nm_SCALE
        # xt_hat=self.backbone_init.sample( chain_idx,Z=z_hat)
        # print(torch.mean(xt_hat-xt))

        return xt * res_mask[..., None, None]

    # def _corrupt_trans(self, trans_1, t,rg, res_mask):
    #     trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
    #     trans_0 = trans_nm_0 * rg
    #     trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
    #     trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
    #     trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
    #     return trans_t * res_mask[..., None]

    def _corrupt_trans(self, trans_1, t,rg,  res_mask,fixed_mask=None,design=False):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, None, self._device)
        trans_0 = trans_nm_0 * rg
        if not design:

            trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        else:
            trans_0 = trans_0
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1

        if fixed_mask is not None:
            diffuse_mask=res_mask * (1-fixed_mask)
            trans_t = _trans_diffuse_mask(trans_t, trans_1,diffuse_mask )
        else:
            trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]


    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask,fixed_mask=None):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch * num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
        )

        if fixed_mask is not None:
            diffuse_mask=res_mask * (1-fixed_mask)
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1,diffuse_mask )
        else:
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

        return rotmats_t

    def _center(self,batch):
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')
        if 'bbatoms' not in batch and 'atoms14' in batch:
            bb_pos = batch['atoms14'][:, :, 1]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'], dim=1) + 1e-5)[:, None]
            batch['atoms14'] = batch['atoms14'] - bb_center[:, None, None, :]

        else:
            bb_pos = batch['bbatoms'][:,:, 1]*batch['res_mask'][...,None]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'],dim=1) + 1e-5)[:,None]
            batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def _center_bbatoms_t(self,batch):
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')

        bb_pos = batch['bbatoms_t'][:,:, 1]*batch['res_mask'][...,None]
        bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['res_mask'],dim=1) + 1e-5)[:,None]
        batch['bbatoms_t'] = batch['bbatoms_t'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch
    def _motif_center(self,batch):
        '''
        only use the center of the fixed area, 这样尝试避免复合物距离过大
        '''
        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center_before.pdb')
        if 'bbatoms' not in batch and 'atoms14' in batch:
            bb_pos = batch['atoms14'][:, :, 1]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum(batch['fixed_mask'], dim=1) + 1e-5)[:, None]
            batch['atoms14'] = batch['atoms14'] - bb_center[:, None, None, :]

        else:
            bb_pos = batch['bbatoms'][:,:, 1]*(batch['fixed_mask'])[...,None]
            bb_center = torch.sum(bb_pos, dim=1) / (torch.sum((batch['fixed_mask']),dim=1) + 1e-5)[:,None]
            batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def _binder_center(self,batch):
        '''
        only use the center of the fixed area, 这样尝试避免复合物距离过大
        '''

        bb_pos = batch['bbatoms'][:,:, 1]*(batch['fixed_mask'])[...,None]
        bb_center = torch.sum(bb_pos, dim=1) / (torch.sum((batch['fixed_mask']),dim=1) + 1e-5)[:,None]
        batch['bbatoms'] = batch['bbatoms'] - bb_center[:, None, None, :]

        #save_pdb(batch['bbatoms'][0,:,:4,:].detach().cpu().numpy().reshape(-1, 3), 'motif_center.pdb')
        return batch

    def corrupt_batch(self, batch):

        # for rcsb should be centered
        batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)
        #scope
        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        #rcsb
        bbatoms = noisy_batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        if 'chain_idx'  not in noisy_batch:
            chain_idx = torch.ones_like(res_mask)
            noisy_batch['chain_idx'] = chain_idx
            noisy_batch['bbatoms'] = self.frame_builder(rotmats_1, trans_1, chain_idx).float()

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = None
        return noisy_batch



    def corrupt_batch_base_ss(self, batch):
        '''

        we add ss info,  and train base model

        '''

        # for rcsb should be centered
        batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)



        #rcsb
        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        fixed_mask=torch.zeros_like(res_mask)
        batch['ss'] = SS_mask(batch['ss'],fixed_mask,design=False)
        noisy_batch['ss'] = batch['ss']







        if 'chain_idx'  not in noisy_batch:
            chain_idx = torch.ones_like(res_mask)
            noisy_batch['chain_idx'] = chain_idx
            noisy_batch['bbatoms'] = self.frame_builder(rotmats_1, trans_1, chain_idx).float()

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = fixed_mask
        return noisy_batch


    def corrupt_seq(self, batch,aug_eps=0.1):

        # for rcsb should be centered
        batch['bbatoms']= batch['atoms4']
        batch=self._center(batch)
        batch['bbatoms']= batch['bbatoms']+torch.randn_like(batch['bbatoms'])*aug_eps

        noisy_batch = copy.deepcopy(batch)
        #scope
        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        #rcsb
        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape


        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = torch.ones_like(t)

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)


        noisy_batch['trans_t'] = trans_1



        noisy_batch['rotmats_t'] = rotmats_1
        noisy_batch['bbatoms_t'] = batch['bbatoms']
        noisy_batch['fixed_mask'] = res_mask
        return noisy_batch
    def fixt_corrupt_batch(self, batch,t):
        # batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)

        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]

        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask)
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)

        xt=self._corrupt_bbatoms(bbatoms,chain_idx, t, res_mask)
        rotmats_t2, trans_t2, _q = self.frame_builder.inverse(xt, chain_idx)  # frames in new rigid system



        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()

        noisy_batch['trans_t2'] = trans_t2
        noisy_batch['rotmats_t2'] = rotmats_t2
        noisy_batch['bbatoms_t2'] = self.frame_builder(rotmats_t2, trans_t2, chain_idx).float()

        noisy_batch['fixed_mask'] = None
        return noisy_batch
    def corrupt_batch_motif(self, batch):
        '''
        这部分扰动坐标，主要是随机生成一个mask，被mask住的部分其trans和rot 数据保持不动，用于训练motif模型
        '''

        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        #make res mask
        motifornot = torch.rand(1, device=self._device)
        #fixed_mask = design_masks(res_mask.shape[0], res_mask.shape[1]).to(self._device)
        #noisy_batch['fixed_mask'] = fixed_mask
        if motifornot > 0.5:
            fixed_mask = design_masks(res_mask.shape[0], res_mask.shape[1]).to(self._device)
            noisy_batch['fixed_mask'] = fixed_mask

            noisy_batch = self._motif_center(noisy_batch)
            # #recenter
            # center_CA = ((fixed_mask[:, :, None]) * bbatoms[..., 1, :]).sum(dim=1) / (
            #         (fixed_mask[:, :, None]).sum(dim=1) + 1e-4)  # (B,  3)
            # bbatoms = bbatoms - center_CA[:, None, None, :]
            # print(noisy_batch['fixed_mask'].sum(dim=-1) / noisy_batch['fixed_mask'].shape[1])
        else:
            fixed_mask = res_mask*0
            noisy_batch['fixed_mask'] = fixed_mask # we do not fixed any pos, model generate all

            noisy_batch=self._center(noisy_batch)


        # sample area set to zero




        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')

        # must after center

        bbatoms = noisy_batch['bbatoms'].float()  # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1



        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        trans_t = self._corrupt_trans(trans_1, t,rg, res_mask,fixed_mask)
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask,fixed_mask)

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        # save_pdb(noisy_batch['bbatoms_t'][0].reshape(-1,3), 'motif_test.pdb')
        # save_pdb(bbatoms[0].reshape(-1,3), 'motif_gt.pdb')


        return noisy_batch





    def corrupt_batch_binder_sidechain(self, batch,precision,t=None,noise=False):
        '''
        这部分主要是用于binder的训练，增加了SS 数据，和毗邻矩阵，且也增加了b-factor的部分，还有chi 角
        noise: when train side, could noise binder area or not
        '''

        batch=self._center(batch)
        if precision == 'fp16':
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].half()
        else:
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].float()



        noisy_batch = copy.deepcopy(batch)
        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        #print(res_mask.shape)
        bbatoms = batch['atoms14'][...,:4,:] # Angstrom

        binder_mask=create_binder_mask(batch['com_idx'],batch['chain_idx']).int()
        noisy_batch['fixed_mask'] = binder_mask
        noisy_batch['bbatoms'] = bbatoms
        fixed_mask=binder_mask


        batch['ss'] = SS_mask(batch['ss'],fixed_mask)


        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1


        if t is None:
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            noisy_batch['t'] = t
        else:
            t_ = self.sample_t(num_batch)[:, None]
            t = (t_*(1-t))+t
            noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)



        if not noise:
            noisy_batch['trans_t'] = trans_1
            noisy_batch['rotmats_t'] = rotmats_1
            noisy_batch['bbatoms_t'] =bbatoms.float()

        else:
            trans_t = self._corrupt_trans(trans_1, t, rg, res_mask, fixed_mask)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask, fixed_mask)
            noisy_batch['trans_t'] = trans_t
            noisy_batch['rotmats_t'] = rotmats_t
            noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()


        save_pdb_chain(noisy_batch['bbatoms_t'][0].reshape(-1,3).cpu().numpy(),chain_idx[0].cpu().numpy(), 'binder_1.pdb')


        return noisy_batch

    def corrupt_batch_DYNbinder(self, batch,precision,binder_mask=None,design_num=0,t=None,design=False,path=None,hotspot=False):
        '''
        这部分主要是用于binder的训练，增加了SS 数据，和毗邻矩阵，且也增加了b-factor的部分，还有chi 角
        binder mask 是要固定的位置

        hotspot  if true, it as center
        '''

        save_pdb_chain(batch['atoms14'][0][:,:4,:].reshape(-1, 3).cpu().numpy(), batch['chain_idx'][0].cpu().numpy(),
                       f'/{path}/native_0_nonoise_{str(design_num)}.pdb')
        save_pdb_chain(batch['atoms14'][1][:,:4,:].reshape(-1, 3).cpu().numpy(), batch['chain_idx'][0].cpu().numpy(),
                       f'/{path}/native_1_nonoise_{str(design_num)}.pdb')

        # if not hotspot:
        #     # use new center
        #     if design:
        #         # use fixed area to compute zeros
        #         batch = self._motif_center(batch)
        #     else:
        #         # batch=self._center(batch)  如果对整体进行中心化，其实会暴露binder的位置，所以应该以target为中心
        #         batch = self._motif_center(batch)


        if precision == 'fp16':
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].half()
        else:
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].float()



        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        # print(res_mask.shape)
        bbatoms = batch['atoms14'][...,:4,:] # Angstrom

        if binder_mask is None:
            binder_mask=create_binder_mask(batch['com_idx'],batch['chain_idx']).int()
        else:
            binder_mask=binder_mask.int()
        noisy_batch['fixed_mask'] = binder_mask
        noisy_batch['bbatoms'] = bbatoms
        fixed_mask=binder_mask

        if not hotspot:
            noisy_batch = self._binder_center(noisy_batch)

        batch['ss'] = SS_mask(batch['ss'],fixed_mask,design)
        noisy_batch['ss'] = batch['ss']

        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(noisy_batch['bbatoms'] , chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1


        if t is None:
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            noisy_batch['t'] = t
        else:
            ## when inference, set t=0,mean start from 1st  t=1 mean fixed not change
            sample_t = self.sample_t(num_batch)[:, None]
            t=torch.ones_like(sample_t)*t
            noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        if not design:
            trans_t = self._corrupt_trans(trans_1, t,rg, res_mask,fixed_mask,design)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask,fixed_mask)
        else:
            trans_0 = _centered_gaussian(
                num_batch, trans_1.shape[1], None, self._device) * rg

            diffuse_mask=res_mask * (1-fixed_mask)
            trans_t = _trans_diffuse_mask(trans_0, trans_1,diffuse_mask )

            rotmats_0 = _uniform_so3(num_batch, trans_1.shape[1], self._device)

            rotmats_t = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()

        # 替换 binder_mask==0 的区域为 batch=0 的数据
        for key in ['atoms14', 'bbatoms', 'trans_1', 'rotmats_1', 'trans_t', 'rotmats_t', 'bbatoms_t']:
            if key in noisy_batch:
                data = noisy_batch[key]  # shape: [B, N, 14, 3] or similar
                mask = binder_mask  # shape: [B, N]

                # Ensure mask is broadcastable to data
                while mask.dim() < data.dim():
                    mask = mask.unsqueeze(-1)  # shape: [B, N, 1] -> [B, N, 1, 1] etc.

                B = data.shape[0]
                for b in range(1, B):
                    # data[b] = where(mask[b] == 0, data[0], data[b])
                    # torch.where(cond, A, B): where cond==True, pick A, else pick B

                    data[b] = torch.where(mask[b].expand(data[0].shape) == 0, data[0], data[b])

                noisy_batch[key] = data

        # noisy_batch=self._center_bbatoms_t(noisy_batch)
        # rotmats_t,trans_t, _q=self.frame_builder.inverse(noisy_batch['bbatoms_t'],chain_idx)
        # noisy_batch['trans_t']=trans_t
        # noisy_batch['rotmats_t'] =rotmats_t

        if path is not None:


            save_pdb_chain(noisy_batch['bbatoms_t'][0].reshape(-1,3).cpu().numpy(),chain_idx[0].cpu().numpy(), f'/{path}/center_{str(design_num)}.pdb')

            # save_pdb_chain(bbatoms[0].reshape(-1, 3).cpu().numpy(), chain_idx[0].cpu().numpy(),
            #                f'{path}/native_x0_{str(design_num)}.pdb')

        return noisy_batch
    def corrupt_batch_binder(self, batch,precision,binder_mask=None,design_num=0,t=None,design=False,path=None,hotspot=False):
        '''
        这部分主要是用于binder的训练，增加了SS 数据，和毗邻矩阵，且也增加了b-factor的部分，还有chi 角
        binder mask 是要固定的位置

        hotspot  if true, it as center
        '''

        # save_pdb_chain(batch['atoms14'][0][:,:4,:].reshape(-1, 3).cpu().numpy(), batch['chain_idx'][0].cpu().numpy(),
        #                f'/{path}/native_nonoise_{str(design_num)}.pdb')

        # if not hotspot:
        #     # use new center
        #     if design:
        #         # use fixed area to compute zeros
        #         batch = self._motif_center(batch)
        #     else:
        #         # batch=self._center(batch)  如果对整体进行中心化，其实会暴露binder的位置，所以应该以target为中心
        #         batch = self._motif_center(batch)


        if precision == 'fp16':
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].half()
        else:
            batch['atoms14_b_factors']=batch['atoms14_b_factors'].float()



        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape
        # print(res_mask.shape)
        bbatoms = batch['atoms14'][...,:4,:] # Angstrom

        if binder_mask is None:
            binder_mask=create_binder_mask(batch['com_idx'],batch['chain_idx']).int()
        else:
            binder_mask=binder_mask.int()
        noisy_batch['fixed_mask'] = binder_mask
        noisy_batch['bbatoms'] = bbatoms
        fixed_mask=binder_mask

        if not hotspot:
            noisy_batch = self._binder_center(noisy_batch)

        batch['ss'] = SS_mask(batch['ss'],fixed_mask,design)
        noisy_batch['ss'] = batch['ss']

        # save_pdb(bbatoms[0].reshape(-1,3), 'motif.pdb')
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(noisy_batch['bbatoms'] , chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1




        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1


        if t is None:
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            noisy_batch['t'] = t
        else:
            ## when inference, set t=0,mean start from 1st  t=1 mean fixed not change
            sample_t = self.sample_t(num_batch)[:, None]
            t=torch.ones_like(sample_t)*t
            noisy_batch['t'] = t

        # Apply corruptions
        # init
        rg = (2 / np.sqrt(3)) * (chain_idx.shape[1]-binder_mask.sum()) ** 0.4
        self.backbone_init._register(stddev_CA=rg, device=self._device)

        if not design:
            trans_t = self._corrupt_trans(trans_1, t,rg, res_mask,fixed_mask,design)
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask,fixed_mask)
        else:
            trans_0 = _centered_gaussian(
                num_batch, trans_1.shape[1], None, self._device) * rg

            diffuse_mask=res_mask * (1-fixed_mask)
            trans_t = _trans_diffuse_mask(trans_0, trans_1,diffuse_mask )

            rotmats_0 = _uniform_so3(num_batch, trans_1.shape[1], self._device)

            rotmats_t = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)

        # def _center(_X):
        #     _X = _X - _X.mean(1, keepdim=True)
        #     return _X
        #
        # trans_t=_center(trans_t)

        trans_t=trans_t-(trans_t*fixed_mask[...,None].float()).mean(1, keepdim=True)

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()



        # noisy_batch=self._center_bbatoms_t(noisy_batch)
        # rotmats_t,trans_t, _q=self.frame_builder.inverse(noisy_batch['bbatoms_t'],chain_idx)
        # noisy_batch['trans_t']=trans_t
        # noisy_batch['rotmats_t'] =rotmats_t

        if path is not None:


            save_pdb_chain(noisy_batch['bbatoms_t'][0].reshape(-1,3).cpu().numpy(),chain_idx[0].cpu().numpy(), f'/{path}/center_{str(design_num)}.pdb')

            # save_pdb_chain(bbatoms[0].reshape(-1, 3).cpu().numpy(), chain_idx[0].cpu().numpy(),
            #                f'{path}/native_x0_{str(design_num)}.pdb')

        return noisy_batch

    def corrupt_batch_atoms(self, batch):
        # batch=self._center(batch)

        noisy_batch = copy.deepcopy(batch)

        # # [B, N, 3]
        # trans_1 = batch['trans_1']  # Angstrom
        #
        # # [B, N, 3, 3]
        # rotmats_1 = batch['rotmats_1']

        bbatoms = batch['bbatoms'].float() # Angstrom
        chain_idx = batch['chain_idx']
        gt_rotmats_1, gt_trans_1, _q = self.frame_builder.inverse(bbatoms, chain_idx)  # frames in new rigid system
        trans_1 = gt_trans_1
        rotmats_1 = gt_rotmats_1


        noisy_batch['trans_1'] = trans_1
        noisy_batch['rotmats_1'] = rotmats_1

        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        # init
        xt = self._corrupt_bbatoms(bbatoms, chain_idx, t, res_mask)
        rotmats_t, trans_t, _q = self.frame_builder.inverse(xt, chain_idx)  # frames in new rigid system

        noisy_batch['trans_t'] = trans_t
        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['bbatoms_t'] = self.frame_builder(rotmats_t, trans_t, chain_idx).float()
        noisy_batch['fixed_mask'] = None
        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / ( 1- t)
        return trans_t + trans_vf * d_t

    def _X_Temp_euler_step(self, d_t, t, trans_1, trans_t,):
        alpha_t=t
        lambda_0=self._cfg.temp

        if lambda_0 is None:
            lambda_0=1
        temp=lambda_0 / (alpha_t ** 2 + (1 - alpha_t ** 2) * lambda_0)

        h=1/t
        g2=(2-2*t)/t
        score=(t*trans_1-trans_t)/(1-t)**2
        vf=h*trans_t+0.5*g2*score*temp

        if torch.any(torch.isinf(vf)):
            print(vf)
            print(score)
            raise ValueError(' inf in vf')



        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + vf * d_t

    def pf_ode_step(self, dt, t,  trans_1, trans_t,eps=1e-5):
        """
        Stable PF-ODE step for R^3 with t: 1 -> 0 (so typically dt < 0).
        Matches your stable implementation's behavior but keeps the PF-ODE form.

        x_{t+dt} = x_t + [ f(x_t,t) - 0.5 * g^2(t) * score_hat * temp ] * dt
        where:
          f(x,t) = + x/t       (since dt < 0; equals -x/t * |dt| ),
          g^2(t) = (2 - 2*t)/t (your schedule; damped near t≈1),
          score_hat = (trans_t - t*trans_1) / (1 - t)^2   # NOTE: flipped sign vs. before
          temp = λ0 / (t^2 + (1 - t^2) * λ0)
        """
        # 1) clamp t for stability
        t = t.clamp(min=eps, max=1 - eps)

        # 2) your temperature reweighting
        lambda_0 = getattr(self._cfg, "temp", 1.0) or 1.0
        temp = lambda_0 / (t ** 2 + (1.0 - t ** 2) * lambda_0)

        # 3) drift f and schedule g^2 consistent with your stable code
        f = trans_t / t  # with dt<0 this equals (-x/t)*|dt|
        g2 = (2.0 - 2.0 * t) / t  # damped near t≈1, ~2/t near t≈0

        # 4) use score_hat = - previous score  ==> aligns with PF-ODE "minus" sign
        score_hat = (trans_t - t * trans_1) / (1.0 - t) ** 2

        # 5) PF-ODE drift (minus sign), but with your schedule/temperature
        drift = f - 0.5 * g2 * score_hat * temp

        # 6) Euler step
        x_next = trans_t + drift * dt

        # optional: numeric guards (clip huge steps, squash NaN/Inf)
        # x_next = torch.nan_to_num(x_next, nan=0.0, posinf=1e6, neginf=-1e6)

        return x_next,drift

    def heun_step_R3(self, dt, t, trans_1, trans_t,  eps=1e-5):
        """
        Heun（改进欧拉）：
          1) 预测: x' = x + drift(x,t)*dt
          2) 校正: drift' = drift(x', t+dt)
          3) 合成: x_next = x + 0.5*(drift + drift')*dt
        """
        # predictor
        _,drift1 = self.pf_ode_step(dt,t, trans_1, trans_t, eps=eps)
        x_pred = trans_t + drift1 * dt

        # corrector
        t2 = (t + dt).clamp(min=eps, max=1 - eps)  # 反向时间
        _,drift2 = self.pf_ode_step(dt,t2, trans_1, x_pred, eps=eps)

        x_next = trans_t + 0.5 * (drift1 + drift2) * dt

        return x_next

    def make_sigma_t_fn(self):
        if self._rots_cfg.sample_schedule == "linear":
            def sigma_t_fn(t):
                eps = 1e-6
                return (1.0 - t).clamp(min=eps)
        elif self._rots_cfg.sample_schedule == "exp":
            k = getattr(self._rots_cfg, "exp_k", 1.0)  # 你之前用的因子
            rate = self._rots_cfg.exp_rate

            def sigma_t_fn(t):
                return torch.exp(-(rate / k) * (1.0 - t)).clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown schedule {self._rots_cfg.sample_schedule}")
        return sigma_t_fn

    def heun_step_SO3(
            self,
            dt, t, R_1, R_t,
            temp: float = 1.0,
            eps: float = 1e-6,
            project: bool = False,  # A/B 对比先关；实际可开
            use_trust_region: bool = True,  # score 不超过主项一定比例
            c: float = 0.3,  # 0.3~0.7；默认0.5
    ):
        """
        SO(3) Heun（右乘指数映射）
          1) omega1 = drift(R_t, t,  σ(t))
          2) R_pred = R_t * Exp(dt * omega1)
          3) omega2 = drift(R_pred, t+dt, σ(t+dt))
          4) omega = 0.5*(omega1 + omega2); R_next = R_t * Exp(dt * omega)
        """
        dt = dt.to(R_1.device)
        t = t.to(R_1.device)
        t1 = t.clamp(min=eps, max=1.0 - eps)
        t2 = (t + dt).clamp(min=eps, max=1.0 - eps)

        # σ(t)（只随时间）：与你训练 schedule 一致
        if self._rots_cfg.sample_schedule == "linear":
            sigma_t1 = (1.0 - t1).clamp(min=eps)
            sigma_t2 = (1.0 - t2).clamp(min=eps)
        elif self._rots_cfg.sample_schedule == "exp":
            rate = self._rots_cfg.exp_rate
            k = getattr(self._rots_cfg, "exp_k", 1.0)
            sigma_t1 = torch.exp(-(rate / k) * (1.0 - t1)).clamp(min=eps)
            sigma_t2 = torch.exp(-(rate / k) * (1.0 - t2)).clamp(min=eps)
        else:
            raise ValueError(f"Unknown schedule {self._rots_cfg.sample_schedule}")

        # 漂移1
        omega1 = so3_utils.so3_omega_drift_time(
            R_t=R_t, R_1=R_1, t=t1,
            exp_rate=self._rots_cfg.exp_rate,
            sigma_t=sigma_t1, temp=temp, eps=eps,
        )

        # （可选）信任域：限制 score 相对主项
        if use_trust_region:
            xi = so3_utils.calc_rot_vf(R_t, R_1)
            f = self._rots_cfg.exp_rate * xi
            corr1 = omega1 - f
            f_norm = torch.norm(f, dim=-1, keepdim=True) + 1e-8
            s_norm = torch.norm(corr1, dim=-1, keepdim=True) + 1e-8
            scale = torch.clamp(c * f_norm / s_norm, max=1.0)
            omega1 = f + scale * corr1

        # 预测
        dR1 = so3_utils.rotvec_to_rotmat(omega1 * dt[..., None])
        R_pred = so3_utils.rot_mult(R_t, dR1)
        if project:
            R_pred = so3_utils.project_to_so3_fast(R_pred)

        # 漂移2（t+dt，用 σ(t+dt)）
        omega2 = so3_utils.so3_omega_drift_time(
            R_t=R_pred, R_1=R_1, t=t2,
            exp_rate=self._rots_cfg.exp_rate,
            sigma_t=sigma_t2, temp=temp, eps=eps,
        )
        if use_trust_region:
            xi2 = so3_utils.calc_rot_vf(R_pred, R_1)
            f2 = self._rots_cfg.exp_rate * xi2
            corr2 = omega2 - f2
            f2_norm = torch.norm(f2, dim=-1, keepdim=True) + 1e-8
            s2_norm = torch.norm(corr2, dim=-1, keepdim=True) + 1e-8
            scale2 = torch.clamp(c * f2_norm / s2_norm, max=1.0)
            omega2 = f2 + scale2 * corr2

        # 合成 + 更新
        omega = 0.5 * (omega1 + omega2)
        import math
        theta_max = math.radians(6.5)
        theta = torch.norm(omega, dim=-1, keepdim=True) * dt.abs()[..., None]
        scale = (theta_max / theta.clamp_min(1e-12)).clamp(max=1.0)
        omega = omega * scale

        dR = so3_utils.rotvec_to_rotmat(omega * dt[..., None])
        R_next = so3_utils.rot_mult(R_t, dR)
        if project:
            R_next = so3_utils.project_to_so3_fast(R_next)

        # with torch.no_grad():
        #
        #     rel = (torch.norm(omega2 - omega1, dim=-1) / (torch.norm(omega1, dim=-1) + 1e-8)).mean()
        #     print("Heun correction rel:", rel.item())
        #
        #     r = (torch.norm(omega1 - f, dim=-1) / (torch.norm(f, dim=-1) + 1e-8)).mean()
        #     print("score/f ratio:", r.item())

        return R_next

    def sde_step(self, dt, t, trans_1, trans_t, eps=1e-5, noise_scale=1.0):
        """
        Reverse SDE step for R^3 with t: 1 -> 0 (so typically dt < 0).

        x_{t+dt} = x_t + [ f(x_t,t) - g^2(t) * score_hat * temp ] * dt  +  g(t) dW
        where (kept consistent with your PF-ODE setup):
          f(x,t)     = + x/t                 # with dt<0 equals (-x/t)*|dt|
          g^2(t)     = (2 - 2*t)/t
          score_hat  = (trans_t - t*trans_1) / (1 - t)^2   # flipped sign vs. original score
          temp       = λ0 / (t^2 + (1 - t^2) * λ0)
          dW ~ N(0, |dt| I) implemented as sqrt(|dt|) * N(0, I)

        noise_scale: extra knob (default 1.0); set <1 to reduce stochasticity, >1 to increase.
        """
        # 1) clamp t for stability
        t = t.clamp(min=eps, max=1 - eps)

        # 2) temperature reweighting (same as your PF-ODE)
        lambda_0 = getattr(self._cfg, "temp", 1.0) or 1.0
        temp = lambda_0 / (t ** 2 + (1.0 - t ** 2) * lambda_0)

        # 3) drift pieces (same schedules as PF-ODE)
        f = trans_t / t  # drift base
        g2 = (2.0 - 2.0 * t) / t  # gentle schedule

        # 4) score with flipped sign (to align with PF-ODE's "minus" form)
        score_hat = (trans_t - t * trans_1) / (1.0 - t) ** 2

        # 5) reverse SDE drift: f - g^2 * score_hat * temp
        drift = f - g2 * score_hat * temp

        # 6) Euler–Maruyama noise term: g * dW, with dW ~ N(0, |dt| I)
        #    -> std = sqrt(g^2 * |dt|) = sqrt(g2) * sqrt(|dt|)
        std = (g2.clamp_min(0.0)).sqrt() * (dt.abs().sqrt()) * noise_scale
        dW = torch.randn_like(trans_t) * std

        # 7) update
        x_next = trans_t + drift * dt + dW
        return x_next

    def _X_Temp_motif_euler_step(self, d_t, t, trans_f,trans_1, trans_t,fixed_mask,theta):
        temp = self._cfg.temp
        h=1/t
        g2=(2-2*t)/t
        score_motif=(fixed_mask[...,None]*(t*trans_f-trans_t)/(1-t)**2)*theta
        score=1*((t*trans_1-trans_t)/(1-t)**2)+ score_motif
        vf=h*trans_t+0.5*g2*score*temp


        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + vf * d_t

    def sde(self, d_t, t, trans_1, trans_t,chain_idx,temp=10):
        h=1/t
        g2=(2-2*t)/t
        score=(t*trans_1-trans_t)/(1-t)**2
        vf=(h*trans_t-g2*score)* d_t

        dw=torch.randn_like(trans_1)
        dx=vf  + torch.sqrt(g2)*dw
        #trans_vf = (trans_1 - trans_t) / ( 1- t)

        return trans_t + dx

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def _pf_ode_drift_SO3(self, d_t, t, rotmats_1, rotmats_t, temp=1.0, eps=1e-6):
        """
        兼容旧接口，但只返回 omega_step（不再用 d_t、不更新矩阵）。
        σ 只按时间计算，与训练 schedule 对齐。
        """
        t_safe = t.clamp(min=eps, max=1.0 - eps)
        if self._rots_cfg.sample_schedule == "linear":
            sigma_t = (1.0 - t_safe).clamp(min=eps)
        elif self._rots_cfg.sample_schedule == "exp":
            rate = self._rots_cfg.exp_rate
            k = getattr(self._rots_cfg, "exp_k", 1.0)
            sigma_t = torch.exp(-(rate / k) * (1.0 - t_safe)).clamp(min=eps)
        else:
            raise ValueError(f"Unknown schedule {self._rots_cfg.sample_schedule}")

        omega_step = so3_utils.so3_omega_drift_time(
            R_t=rotmats_t,
            R_1=rotmats_1,
            t=t_safe.to(rotmats_1.device),
            exp_rate=self._rots_cfg.exp_rate,
            sigma_t=sigma_t.to(rotmats_1.device),
            temp=temp,
            eps=eps,
        )
        return omega_step

    def _rots_motif_euler_step(self, d_t, t, rotmats_f,rotmats_1,rotmats_t,fixed_mask,theta):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')

        mid_rot=so3_utils.geodesic_t(
            theta, rotmats_f, rotmats_1)

        mid_rot=rotmats_1* (1 - fixed_mask[...,None,None]) +  mid_rot * fixed_mask[...,None,None]
        return so3_utils.geodesic_t(
            scaling* d_t, mid_rot, rotmats_t)

    def elbo(self, X0_pred, X0, C, t, loss_mask):
        """ITD ELBO as a weighted average of denoising error,
        inspired by https://arxiv.org/abs/2302.03792"""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(X0.device)

        # Interpolate missing data with Brownian Bridge posterior
        # X0 = backbone.impute_masked_X(X0, C)
        # X0_pred = backbone.impute_masked_X(X0_pred, C)

        # Compute whitened residual

        X0 = X0 * loss_mask[..., None, None]
        X0_pred = X0_pred * loss_mask[..., None, None]
        dX = (X0 - X0_pred).reshape([X0.shape[0], -1, 3])
        R_inv_dX = self.backbone_init._multiply_R_inverse(dX, C)

        # Average per atom, including over "missing" positions that we filled in
        weight = 0.5 * self.noise_perturb.SNR_derivative(t)[:, None]
        snr = self.noise_perturb.SNR(t)[:, None]

        c = R_inv_dX.pow(2)
        v = 1 / (1 + snr)

        # Compute per-atom loss

        # loss_itd = (
        #     weight * (R_inv_dX.pow(2) )
        #     - 0.5 * np.log(np.pi * 2.0 * np.e)
        # ).reshape(X0.shape)

        # if minus 1/1+snr  the lossidt could be zheng
        loss_itd = (
                weight * (R_inv_dX.pow(2) - 1 / (1 + snr))
                - 0.5 * np.log(np.pi * 2.0 * np.e)
        ).reshape(X0.shape)

        # Compute average per-atom loss (including over missing regions)
        mask = loss_mask.float()
        mask_atoms = mask.reshape(mask.shape + (1, 1)).expand([-1, -1, 4, 1])

        # Per-complex
        elbo_gap = (mask_atoms * loss_itd).sum([1, 2, 3])
        logdet = self.backbone_init.log_determinant(C)
        elbo_unnormalized = elbo_gap   - logdet

        # Normalize per atom
        elbo = elbo_unnormalized / (mask_atoms.sum([1, 2, 3]) + self._eps)

        # Compute batch average
        weights = mask_atoms.sum([1, 2, 3])
        elbo_batch = (weights * elbo).sum() / (weights.sum() + self._eps)
        mmse= (c).sum() / (weights.sum() + self._eps)
        return elbo, elbo_batch
    def pseudoelbo(self, loss_per_residue, C, t):
        """Compute pseudo-ELBOs as weighted averages of other errors."""
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(C.device)
        self._eps=1e-5
        # Average per atom, including over x"missing" positions that we filled in
        weight = 0.5 * self.noise_perturb.SNR_derivative(t)[:, None]
        loss = weight * loss_per_residue

        # Compute average loss
        mask = (C > 0).float()
        pseudoelbo = (mask * loss).sum(-1) / (mask.sum(-1) + self._eps)
        pseudoelbo_batch = (mask * loss).sum() / (mask.sum() + self._eps)
        return pseudoelbo, pseudoelbo_batch
    def _loss_pseudoelbo(self,  X0_pred, X, C, t, w=None, X_t_2=None):
        # Unaligned residual pseudoELBO
        self.loss_scale=10
        unaligned_mse = ((X - X0_pred) / self.loss_scale).square().sum(-1).mean(-1)
        elbo_X, batch_pseudoelbo_X = self.pseudoelbo(
            unaligned_mse, C, t
        )
        return elbo_X,batch_pseudoelbo_X

    def _se3_loss(self,noisy_batch,model_output,training_cfg,_exp_cfg):

        loss_mask = noisy_batch['res_mask']
        num_batch, num_res = loss_mask.shape
        # Ground truth labels
        gt_trans_0 = noisy_batch['trans_1']
        gt_rotmats_0 = noisy_batch['rotmats_1']
        gt_bb_atoms = noisy_batch['bbatoms']
        chain_idx = noisy_batch['chain_idx']

        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t.type(torch.float32), gt_rotmats_0.type(torch.float32))


        # Timestep used for normalization.
        # in frameflow t~1, is same to ~ gt, normscale~0, so normscale ~ 1-t
        # in ours t~0, is same to ~ gt, normscale~0, so normscale=t
        t = noisy_batch['t']
        norm_scale =  torch.max(
            t[..., None], 1-torch.tensor(training_cfg.t_normalize_clip))



        # Model output predictions.
        pred_trans_0 = model_output['pred_trans']
        pred_rotmats_0 = model_output['pred_rotmats']


        # Backbone atom loss
        pred_bb_atoms = self.frame_builder(pred_rotmats_0, pred_trans_0, chain_idx)
        #vio_loss = self._loss_vio(loss_mask, pred_bb_atoms, noisy_batch['aatype'], noisy_batch['res_idx'], norm_scale)

        # Loss=RL( pred_bb_atoms,gt_bb_atoms,noisy_batch['bbatoms_t'],chain_idx,t,loss_mask)
        # elbo


        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom
        bb_atom_loss=bb_atom_loss

        # Translation VF loss
        trans_error = (gt_trans_0 - pred_trans_0) / norm_scale * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf ) / norm_scale
        R_ij_error = ((gt_rotmats_0 - pred_rotmats_0).square().sum([-1, -2]) * loss_mask).sum(-1)
        # rots_vf_error =  so3_utils.calc_rot_vf(
        #     gt_rotmats_1.type(torch.float32),pred_rotmats_1.type(torch.float32))/ norm_scale
        if torch.any(torch.isnan(rots_vf_error)):
            print('gt_rotmats_1:', gt_rotmats_0)
            print('pred_rotmats_1:', pred_rotmats_0)
            print(torch.mean(gt_rotmats_0 - pred_rotmats_0))

        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        rots_vf_loss = rots_vf_loss #+ R_ij_error / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 4, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 4, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 4])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 4])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        dist_mat_loss=dist_mat_loss

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss)* (
                t> training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= _exp_cfg.training.aux_loss_weight
        se3_vf_loss += auxiliary_loss
        se3_vf_loss = se3_vf_loss #+ vio_loss

        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "R_ij_error": R_ij_error / loss_denom,

        }

    def sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)
        res_idx=torch.arange(num_res,device=self._device)[None]
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_idx':res_idx ,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask':res_mask*0,
            'ss':torch.zeros_like(res_mask).long(),}

        # Set-up time
        # step=np.clip(round(self._sample_cfg.Scaling_coefficient * num_res**self._sample_cfg.Scaling_exponent), min=50, max=1200)
        step=steps_by_length(num_res)

        # if num_res<100:
        #     step=200
        # elif 100<=num_res<200:
        #     step=500
        # elif 200<=num_res<250:
        #     step=200

        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        # gamma = 2.0
        # ts = 1.0 - (1.0 - ts) ** gamma

        # ts = self._cfg.min_t + (1.0 - self._cfg.min_t) * (1 - torch.cos(
        #     ((torch.arange(self._sample_cfg.num_timesteps,
        #                    dtype=torch.float32) / self._sample_cfg.num_timesteps) + 0.008)
        #     / (1 + 0.008) * np.pi / 2
        # ) ** 2)


        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        print(f'Running {self._sample_cfg.num_timesteps} timesteps')
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()

    def heun_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)
        res_idx=torch.arange(num_res,device=self._device)[None]
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_idx':res_idx ,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask':res_mask*0,
            'ss':torch.zeros_like(res_mask).long(),}

        # Set-up time
        # step=np.clip(round(self._sample_cfg.Scaling_coefficient * num_res**self._sample_cfg.Scaling_exponent), min=50, max=1200)
        step=steps_by_length(num_res)

        # if num_res<100:
        #     step=200
        # elif 100<=num_res<200:
        #     step=500
        # elif 200<=num_res<250:
        #     step=200

        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        # gamma = 2.0
        # ts = 1.0 - (1.0 - ts) ** gamma

        # ts = self._cfg.min_t + (1.0 - self._cfg.min_t) * (1 - torch.cos(
        #     ((torch.arange(self._sample_cfg.num_timesteps,
        #                    dtype=torch.float32) / self._sample_cfg.num_timesteps) + 0.008)
        #     / (1 + 0.008) * np.pi / 2
        # ) ** 2)


        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        print(f'Running {self._sample_cfg.num_timesteps} timesteps')
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self.heun_step_R3(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self.heun_step_SO3(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()

    def try_motif_sample(
            self,
            num_batch,
            num_res: list,
            model,

            chain_idx,
            native_X,
            mode='motif',
            fixed_mask=None,
            training=False,
    ):

        num_res=num_res[0]
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)
        res_idx=torch.arange(num_res,device=self._device)[None]
        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0



        batch = {
            'res_idx':res_idx ,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask':fixed_mask,
            'ss':torch.zeros_like(res_mask).long(),
            'bbatoms':native_X,}


        motif_X = self._motif_center(batch)['bbatoms']


        rotmats_m,trans_m, _=self.frame_builder.inverse(motif_X, chain_idx)

        # fix motif area
        diffuse_mask=res_mask*(1-fixed_mask)
        trans_0 = _trans_diffuse_mask(trans_0, trans_m, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_m, diffuse_mask)




        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        print(f'Running {self._sample_cfg.num_timesteps} timesteps')
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()


    def hybrid_sample(
            self,
            num_batch,
            num_res,
            model,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        chain_idx = torch.ones(num_batch, num_res, device=self._device)

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4

        self.backbone_init._register(stddev_CA=rg, device=res_mask.device)
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'fixed_mask': None,}

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X
            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)


            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2


        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        S=torch.ones_like(chain_idx)
        return atom37_traj.detach().cpu(),chain_idx.detach().cpu(),S.detach().cpu()



    def _init_atoms_backbone(self,num_batch, num_res,chain_idx):

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg
        trans_z = trans_0 / rg  # nm _SCALE

        # 2. sample N C O and use batchot to solve it to get others_t
        num_batch = chain_idx.shape[0]
        num_residues = chain_idx.shape[1]


        z = torch.rand(num_batch, num_residues, 4, 3).to(trans_0.device)
        mask = torch.zeros(num_batch, num_residues, 4, ).to(trans_0.device)
        mask[..., 1] = 1
        others_z = z * (1 - mask[..., None])  ##nm _SCALE

        # 3. combine and transform to resgas
        self.backbone_init._register(stddev_CA=rg, device=self._device)
        z = mask[..., None] * trans_z.unsqueeze(-2).repeat(1, 1, 4, 1) + others_z
        bbatoms = self.backbone_init.sample(chain_idx, Z=z) # nm_SCALE
        return bbatoms

    def _init_complex_backbone(self,num_batch, num_res,rgadd=1):

        chain_idx = torch.cat(
            [torch.full([rep], i+1) for i, rep in enumerate(num_res)]
        ).to(self._device).expand(num_batch, -1)
        res_mask = torch.ones_like(chain_idx)

        num_res=chain_idx.shape[1]

        rg = (2 / np.sqrt(3)) * chain_idx.shape[1] ** 0.4
        rg=rg*rgadd
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, None, self._device) * rg

        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)

        return res_mask, chain_idx, trans_0, rotmats_0,rg

    def _align_prediction_to_motif_per_chain(self, pred_trans, pred_rotmats, trans_motif, rotmats_motif, chain_idx, fixed_mask):
        """
        Align each chain's prediction to its original motif using Kabsch algorithm on atom positions.

        For each chain:
        1. Extract motif atom positions (pred vs orig)
        2. Use Kabsch to find optimal R and t: orig = pred @ R + t
        3. Apply (R, t) to entire chain

        Args:
            pred_trans: Predicted translations [B, L, 3]
            pred_rotmats: Predicted rotations [B, L, 3, 3]
            trans_motif: Original motif translations [B, L, 3]
            rotmats_motif: Original motif rotations [B, L, 3, 3]
            chain_idx: Chain indices [B, L]
            fixed_mask: Motif mask [B, L]

        Returns:
            aligned_trans: Aligned translations [B, L, 3]
            aligned_rotmats: Aligned rotations [B, L, 3, 3]
        """
        from data.utils import align_structures

        device = pred_trans.device
        num_batch = pred_trans.shape[0]
        aligned_trans = pred_trans.clone()
        aligned_rotmats = pred_rotmats.clone()

        # Get unique chains
        unique_chains = torch.unique(chain_idx[0])

        for chain_id in unique_chains:
            # Get mask for this chain
            chain_mask = (chain_idx[0] == chain_id)  # [L]

            # Get motif mask for this chain
            motif_mask_chain = fixed_mask[0, chain_mask].bool()  # [chain_len]

            if not motif_mask_chain.any():
                # No motif in this chain, skip alignment
                continue

            # Extract motif atom positions for this chain
            pred_trans_chain = pred_trans[:, chain_mask, :]  # [B, chain_len, 3]
            trans_motif_chain = trans_motif[:, chain_mask, :]  # [B, chain_len, 3]

            pred_motif_atoms = pred_trans_chain[:, motif_mask_chain, :]  # [B, n_motif, 3]
            orig_motif_atoms = trans_motif_chain[:, motif_mask_chain, :]  # [B, n_motif, 3]

            n_motif = pred_motif_atoms.shape[1]
          #  print(f'[Align] Chain {chain_id.item()}: Using Kabsch on {n_motif} motif atoms')

            # Flatten for align_structures: [B*n_motif, 3]
            pred_flat = pred_motif_atoms.reshape(-1, 3)
            orig_flat = orig_motif_atoms.reshape(-1, 3)
            batch_indices = torch.arange(num_batch, device=device).repeat_interleave(n_motif)

            # Use Kabsch algorithm to get rotation matrix
            _, _, R = align_structures(pred_flat, batch_indices, orig_flat)

            # Step 1: Rotate pred_motif_atoms first
            pred_motif_rotated = torch.einsum('bnc,bcd->bnd', pred_motif_atoms, R.float())  # [B, n_motif, 3]

            # Step 2: Check difference for EACH atom (not mean!)
            diff_per_atom = orig_motif_atoms - pred_motif_rotated  # [B, n_motif, 3]

            # print(f'[Align] Chain {chain_id.item()}: After rotation, per-atom differences:')
            # for i in range(min(n_motif, 10)):  # Show first 10 atoms
            #     diff_vec = diff_per_atom[0, i].cpu().numpy()
            #     dist = torch.norm(diff_per_atom[0, i]).item()
            #     print(f'  Atom {i}: diff={diff_vec}, distance={dist:.3f} Å')

            # Step 3: Compute translation as mean of differences
            t = diff_per_atom.mean(dim=1, keepdim=True)  # [B, 1, 3]

            # print(f'  Mean translation needed: {t.squeeze().cpu().numpy()}')
            # print(f'  Rotation trace: {R[0].trace().item():.3f}')

            # Apply transformation to entire chain: aligned = trans @ R + t
            chain_trans_full = pred_trans[:, chain_mask, :]  # [B, chain_len, 3]
            chain_rotmats_full = pred_rotmats[:, chain_mask, :, :]  # [B, chain_len, 3, 3]

            chain_len = chain_trans_full.shape[1]

            # Transform translations: trans_aligned = trans @ R + t
            # [B, chain_len, 3] @ [B, 3, 3] -> [B, chain_len, 3]
            trans_aligned = torch.einsum('blc,bcd->bld', chain_trans_full, R.float()) + t

            # Transform rotations: rotmats_aligned = R @ rotmats
            # [B, 3, 3] @ [B, chain_len, 3, 3] -> [B, chain_len, 3, 3]
            rotmats_aligned = torch.einsum('bik,bljk->blij', R.float(), chain_rotmats_full)

            # Update aligned tensors
            aligned_trans[:, chain_mask, :] = trans_aligned
            aligned_rotmats[:, chain_mask, :, :] = rotmats_aligned

        return aligned_trans, aligned_rotmats

    def _init_per_chain_around_motif(self, num_batch, chain_idx, fixed_mask, trans_motif, rotmats_motif, rgadd=1.0):
        """
        Initialize each chain separately, centered around its own motif region.

        Args:
            num_batch: Batch size
            chain_idx: Chain indices [B, L]
            fixed_mask: Motif mask [B, L] (1 = motif, 0 = scaffold)
            trans_motif: Motif translations [B, L, 3]
            rotmats_motif: Motif rotations [B, L, 3, 3]
            rgadd: Radius of gyration multiplier

        Returns:
            trans_0: Initial translations [B, L, 3]
            rotmats_0: Initial rotations [B, L, 3, 3]
        """
        device = chain_idx.device
        total_len = chain_idx.shape[1]

        # Get unique chain IDs
        unique_chains = torch.unique(chain_idx[0])
        print(f'[Init] Initializing {len(unique_chains)} chains separately')

        # Initialize output tensors
        trans_0 = torch.zeros(num_batch, total_len, 3, device=device)
        rotmats_0 = torch.zeros(num_batch, total_len, 3, 3, device=device)

        for chain_id in unique_chains:
            # Get mask for this chain
            chain_mask = (chain_idx[0] == chain_id)  # [L]
            chain_indices = torch.where(chain_mask)[0]
            chain_len = chain_indices.shape[0]

            # Get motif mask for this chain
            motif_mask_chain = fixed_mask[0, chain_mask].bool()  # [chain_len]

            if motif_mask_chain.any():
                # Calculate motif center for this chain
                motif_trans_chain = trans_motif[:, chain_mask, :]  # [B, chain_len, 3]
                motif_center = motif_trans_chain[:, motif_mask_chain, :].mean(dim=1, keepdim=True)  # [B, 1, 3]
                print(f'[Init] Chain {chain_id.item()}: {chain_len} residues, {motif_mask_chain.sum().item()} motif residues, center={motif_center.squeeze().cpu().numpy()}')
            else:
                # No motif in this chain, use zero center
                motif_center = torch.zeros(num_batch, 1, 3, device=device)
                print(f'[Init] Chain {chain_id.item()}: {chain_len} residues, no motif')

            # Calculate rg for this chain
            rg = (2 / np.sqrt(3)) * chain_len ** 0.4 * rgadd

            # Initialize random positions around motif center for this chain
            trans_chain = _centered_gaussian(num_batch, chain_len, None, device) * rg + motif_center  # [B, chain_len, 3]
            rotmats_chain = _uniform_so3(num_batch, chain_len, device)  # [B, chain_len, 3, 3]

            # Place into output tensors
            trans_0[:, chain_mask, :] = trans_chain
            rotmats_0[:, chain_mask, :, :] = rotmats_chain

        return trans_0, rotmats_0
    def hybrid_Complex_sample(
            self,
            num_batch,
            num_res,

            model,
            ss=None,
    ):

        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)

        trans_0 = trans_0
        rotmats_0 = rotmats_0

        if ss is None:
            ss = res_mask

        batch = {
            'res_mask': res_mask,
            'fixed_mask': torch.zeros_like(res_mask),
            'chain_idx': chain_idx,
            'ss': torch.tensor(ss).to(res_mask.device).unsqueeze(0),
                }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X=atom37_traj
        C=chain_idx+25
        S=torch.ones_like(C)
        return X,C,S


    def hybrid_Complex_sym_sample(
            self,
            num_batch,
            num_res,
            # ss,
            model,
            symmetry='c4',
            recenter=True,
            radius=0,
    ):

        self.symmetry=SymGen(
                symmetry,
                recenter,
                radius,
            )


        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)

        idx_pdb=torch.arange(num_res[0]).long().to(chain_idx.device).unsqueeze(0).repeat(chain_idx.shape[0], 1)
        idx_pdb, chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        chains=[]
        from data.utils import chain_str_to_int,CHAIN_TO_INT
        for i in chain_idx:
            chains.append(int(CHAIN_TO_INT[i])-25)


        chain_idx=torch.tensor(chains).to(res_mask.device).unsqueeze(0).repeat(res_mask.shape[0], 1)
        #intialize S
        S = torch.ones_like(chain_idx)


        trans_0 = trans_0
        rotmats_0 = rotmats_0

        batch = {
            'res_mask': res_mask,
            'fixed_mask': None,
            'chain_idx': chain_idx,
            'ss': torch.zeros_like(res_mask),
                }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take forward step
            bb_atoms_pred = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            sym_bb_atoms_pred,_ = self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))
            sym_bb_atoms_pred=sym_bb_atoms_pred.to(res_mask.device)
            pred_rotmats_1, pred_trans_1, q=self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            # Apply symmetry after denoise
            bb_atoms_pred = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            sym_bb_atoms_pred ,_= self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))



            sym_bb_atoms_pred = sym_bb_atoms_pred.to(res_mask.device)
            rotmats_t_2, trans_t_2, q=self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)



            p = Protein.from_XCS(sym_bb_atoms_pred.unsqueeze(0), chain_idx, res_mask, )
            # p.to_PDB('sym_native_test'+str(t_1)+'.pdb')



            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )

        # Apply symmetry after denoise
        bb_atoms_pred = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
        sym_bb_atoms_pred,_ =  self.symmetry.apply_symmetry(bb_atoms_pred.squeeze(0).to('cpu'), S.squeeze(0).to('cpu'))
        sym_bb_atoms_pred = sym_bb_atoms_pred.to(res_mask.device)
        pred_rotmats_1, pred_trans_1, q = self.frame_builder.inverse(sym_bb_atoms_pred.unsqueeze(0),chain_idx)


        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj
        C=chain_idx+25

        return X,C,S


    def hybrid_Complex_sample_bybinder(
            self,
            num_batch,
            num_res,
            model,
    ):

        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)
        length_res=res_mask.shape[1]
        trans_0 = trans_0
        rotmats_0 = rotmats_0

        fixed_mask=torch.zeros_like(res_mask)

        batch = {
            'res_mask': res_mask,
            'fixed_mask': fixed_mask,
            'chain_idx': chain_idx,

            'ss': torch.zeros_like(res_mask).to(torch.long),
            'aatype': torch.ones_like(res_mask).to(torch.long)*20,
            'chi': torch.zeros(size=(num_batch, length_res,4), device=self._device),
            'mask_chi': torch.zeros(size=(num_batch, length_res,4), device=self._device),

            'res_idx': torch.range(0, length_res - 1, device=self._device).unsqueeze(0),

            'atoms14_b_factors': torch.zeros(size=(num_batch, length_res,4), device=self._device),

                }

        # Set-up time
        # ts = torch.linspace(
        #     self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        ts = torch.linspace(
            1e-2, 1.0, 100)


        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in tqdm(ts[1:]):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch, recycle=1, is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']

            # Take reverse step
            d_t = t_2 - t_1

            # trans_t_2 = self._X_Temp_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            # atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # saved_path = str(t)+'_.pdb'
            # p.to_PDB(saved_path)

            # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        # return [atom37_traj],None

        X = atom37_traj.detach().cpu()
        C = chain_idx.detach().cpu()
        S = model_out['SEQ'].detach().cpu()
        bf = model_out['pred_bf'] * 20

        return X, C, S, bf

    # def hybrid_binder_side_sample(
    #         self,
    #         model,
    #         batch,
    #         sidechain,
    #
    # ):
    #     num_batch = batch['fixed_mask'].shape[0]
    #     chain_idx = batch['chain_idx']
    #     trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
    #     prot_traj = [(trans_0, rotmats_0)]
    #
    #     # Set-up time
    #     ts = torch.linspace(
    #         self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
    #     t_1 = ts[0]
    #
    #     for t_2 in tqdm(ts[1:]):
    #
    #         trans_t_1, rotmats_t_1 = prot_traj[-1]
    #         batch['trans_t'] = trans_t_1
    #         batch['rotmats_t'] = rotmats_t_1
    #         t = torch.ones((num_batch, 1), device=self._device) * t_1
    #         batch['t'] = t
    #         with torch.no_grad():
    #             model_out = model(batch, recycle=1, is_training=True)
    #
    #
    #
    #         # Process model output.
    #         pred_trans_1 = model_out['pred_trans']
    #         pred_rotmats_1 = model_out['pred_rotmats']
    #
    #         # Take reverse step
    #         d_t = t_2 - t_1
    #
    #         trans_t_2 = self._X_Temp_euler_step(
    #             d_t, t_1, pred_trans_1, trans_t_1)
    #
    #         # trans_t_2 = self._trans_euler_step(
    #         #     d_t, t_1, pred_trans_1, trans_t_1)
    #
    #         def _center(_X):
    #             _X = _X - _X.mean(1, keepdim=True)
    #             return _X
    #
    #         trans_t_2 = _center(trans_t_2)
    #         if self._cfg.self_condition:
    #             batch['trans_sc'] = trans_t_2
    #
    #         rotmats_t_2 = self._rots_euler_step(
    #             d_t, t_1, pred_rotmats_1, rotmats_t_1)
    #
    #         prot_traj.append((trans_t_2, rotmats_t_2))
    #         t_1 = t_2
    #
    #
    #
    #     # We only integrated to min_t, so need to make a final step
    #     t_1 = ts[-1]
    #     trans_t_1, rotmats_t_1 = prot_traj[-1]
    #     batch['trans_t'] = trans_t_1
    #     batch['rotmats_t'] = rotmats_t_1
    #     batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
    #
    #     batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
    #     with torch.no_grad():
    #         model_out = model(batch)
    #     pred_trans_1 = model_out['pred_trans']
    #     pred_rotmats_1 = model_out['pred_rotmats']
    #
    #     prot_traj.append((pred_trans_1, pred_rotmats_1))
    #
    #     # Convert trajectories to atom37.
    #     # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
    #     atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()
    #
    #     print('\n now is designing side chain...')
    #     # sidechain
    #     batch['trans_t'] = pred_trans_1
    #     batch['rotmats_t'] = pred_rotmats_1
    #     batch['bbatoms_t'] = atom37_traj
    #
    #     with torch.no_grad():
    #         sidemodel_out = sidechain(batch)
    #     pred_trans_1 = sidemodel_out['pred_trans']
    #     pred_rotmats_1 = sidemodel_out['pred_rotmats']
    #
    #     prot_traj.append((pred_trans_1, pred_rotmats_1))
    #
    #     # Convert trajectories to atom37.
    #     # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
    #     atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()
    #
    #     X = atom37_traj.detach().cpu()
    #     C = chain_idx.detach().cpu()
    #     S = sidemodel_out['SEQ'].detach().cpu()
    #     bf = model_out['pred_bf'] * 20
    #
    #     return X, C, S, bf

    def hybrid_motif_sample(
            self,
            num_batch,
            num_res:list,
            model,

            chain_idx,
            native_X,
            rgadd=1.1,
            ss= None,
            fixed_mask=None,
            training=False,
            sampling_method='heun',  # 'euler' or 'heun'

    ):



        res_mask, _, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res,rgadd)

        #center motif
        motif_X = native_X



        motif_batch = {
            'res_mask': res_mask*fixed_mask,
            'chain_idx': chain_idx,
            'bbatoms':motif_X,
            'fixed_mask': fixed_mask
        }


        motif_X = self._motif_center(motif_batch)['bbatoms']

        # p = Protein.from_XCS(native_X, chain_idx, chain_idx, )
        # p.to_PDB('motif_native_'+str('test_')+'.pdb')
        # p = Protein.from_XCS(motif_X, chain_idx, chain_idx, )
        # p.to_PDB('motif_native_'+str('motif_X')+'.pdb')


        rotmats_m,trans_m, _=self.frame_builder.inverse(motif_X, chain_idx)

        # fix motif area
        diffuse_mask=res_mask*(1-fixed_mask)
        trans_0 = _trans_diffuse_mask(trans_0, trans_m, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_m, diffuse_mask)

        motif_t0 = self.frame_builder(rotmats_0.float(), trans_0, chain_idx)
        p = Protein.from_XCS(motif_t0, chain_idx, chain_idx, )
        p.to_PDB('motif_native_'+str('motif_t0')+'.pdb')

        if ss is not None:
            batch = {
                'res_mask': res_mask,
                'chain_idx': chain_idx,
                'fixed_mask': fixed_mask,
                'ss': ss
            }
        else:
            batch = {
                'res_mask': res_mask,
                'chain_idx': chain_idx,
            'fixed_mask': fixed_mask}



        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            # p = Protein.from_XCS(bb_atoms_t, chain_idx, chain_idx, )
            # p.to_PDB('motif_native_test'+str(t_1)+'.pdb')


            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # #######test motif fixed area
            # bb_atoms_tss = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            # fixed_area_n = native_X.cpu() * fixed_mask[..., None, None].cpu()
            # fixed_area_p = bb_atoms_tss.cpu() * fixed_mask[..., None, None].cpu()
            #
            # RMSD = torch.sum((fixed_area_n - fixed_area_p) ** 2, dim=-1)
            # RMSD = torch.sqrt(RMSD).mean()
            # print('gnnout: ',RMSD)


            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )

            # Take reverse step
            d_t = t_2 - t_1

            # Choose sampling method: euler or heun
            if sampling_method == 'heun':
                trans_t_2 = self.heun_step_R3(
                    d_t, t_1, pred_trans_1, trans_t_1)
            else:  # 'euler'
                trans_t_2 = self._trans_euler_step(
                    d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            # trans_t_2 = self._motif_center()#_center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            if sampling_method == 'heun':
                rotmats_t_2 = self.heun_step_SO3(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)
            else:  # 'euler'
                rotmats_t_2 = self._rots_euler_step(
                    d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        #batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        batch['trans_t'] = pred_trans_1
        batch['rotmats_t'] = pred_rotmats_1
        batch['bbatoms_t'] = atom37_traj




        if training:

            return [atom37_traj],_
        else:
            X=atom37_traj
            C=chain_idx+25
            S=torch.ones_like(C)

            return X, C, S
        #
        #
        # fixed_area_n=native_X.cpu()*fixed_mask[...,None,None].cpu()
        # fixed_area_p = X * fixed_mask[..., None, None].cpu()
        # print('fixed area:  ',fixed_area_p.shape)
        # fixed_area_p=self._batch_ot(fixed_area_p[...,1,:],fixed_area_n[...,1,:],fixed_mask.cpu())
        # print('fixed fixed_area_p:  ',fixed_area_p.shape)
        #
        # RMSD=torch.sum((fixed_area_n[...,1,:]-fixed_area_p)**2,dim=-1)
        # RMSD=torch.sqrt(RMSD.mean(), )
        # print('RMSD:  ',RMSD)
        # return [atom37_traj],_

    def hybrid_motif_sym_sample(
            self,
            num_batch,
            num_res: list,
            model,
            chain_idx,
            native_X,
            fixed_mask=None,
            symmetry='c3',
            recenter=True,
            radius=0.0,
            rgadd=1.0,
            ss=None,
            sampling_method='heun',
            save_traj_dir=None,
            save_interval=10,
    ):
        """
        Hybrid sampling with both motif scaffolding and symmetry constraints.

        Args:
            num_batch: Batch size (usually 1)
            num_res: List of residue counts per chain [asymmetric_unit_length]
            model: Flow model
            chain_idx: Chain indices for asymmetric unit
            native_X: Native backbone atoms for motif region [1, L, 4, 3]
            fixed_mask: Mask indicating motif positions [1, L]
            symmetry: Symmetry type ('c3', 'c4', 'd2', etc.)
            recenter: Whether to recenter subunits
            radius: Radius for recentering
            rgadd: Radius of gyration multiplier
            ss: Secondary structure constraints (optional)
            sampling_method: 'euler' or 'heun'

        Returns:
            X: Backbone coordinates [1, L_total, 4, 3]
            C: Chain indices [1, L_total]
            S: Sequence placeholder [1, L_total]
        """
        from tqdm import tqdm

        # Initialize symmetry generator
        self.symmetry = SymGen(symmetry, recenter, radius)
        print(f'[Motif+Sym] Using {symmetry} symmetry with order {self.symmetry.order}')

        # native_X is already the FULL structure with all chains (e.g., 340 residues for 4 chains)
        # num_res[0] is the total length (already multiplied by order)
        full_len = num_res[0]
        asym_len = full_len // self.symmetry.order
        print(f'[Motif+Sym] Full structure: {full_len} residues ({asym_len} × {self.symmetry.order})')

        # Extract motif frames directly from native_X
        # native_X already contains all chains in correct symmetric positions
        rotmats_m_full, trans_m_full, _ = self.frame_builder.inverse(native_X, chain_idx)

        # Calculate overall motif center and center ALL motifs at origin
        motif_mask_bool = fixed_mask.squeeze(0).bool()  # [L]
        overall_motif_center = trans_m_full[:, motif_mask_bool].mean(1, keepdim=True)  # [B, 1, 3]
        print(f'[Motif+Sym] Overall motif center: {overall_motif_center.squeeze().cpu().numpy()}')

        # Center all motifs at origin
        trans_m_full = trans_m_full - overall_motif_center
        print(f'[Motif+Sym] All motifs centered at origin')

        # Initialize each chain separately around its own motif
        print('[Motif+Sym] Initializing each chain around its own motif...')
        trans_0, rotmats_0 = self._init_per_chain_around_motif(
            num_batch, chain_idx, fixed_mask, trans_m_full, rotmats_m_full, rgadd
        )

        # Create res_mask
        res_mask = torch.ones(num_batch, full_len, device=self._device)

        # S for symmetry application
        S_expanded = torch.ones(full_len)

        # Prepare diffuse mask (1 where we can modify, 0 where motif is fixed)
        diffuse_mask = 1 - fixed_mask

        # Apply motif constraints to initial state
        # Motif regions should NOT have random noise
        trans_0 = _trans_diffuse_mask(trans_0, trans_m_full, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_m_full, diffuse_mask)
        print('[Motif+Sym] Initial state: motif regions fixed, scaffold regions random')

        # Setup batch
        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,  # chain_idx is already full structure
            'fixed_mask': None,
            'ss': torch.zeros_like(res_mask),
        }

        # Setup time steps
        ts = torch.linspace(self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []

        print(f'[Motif+Sym] Sampling {self._sample_cfg.num_timesteps} steps...')
        step_idx = 0
        # for t_2 in tqdm(ts[1:], desc='Sampling', ncols=100):
        for t_2 in ts[1:]:
            step_idx += 1
            # Current state (full symmetric structure)
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            batch['bbatoms_t'] = bb_atoms_t

            # # Save current state for inspection (only chainA, every 10 steps or first 3 steps)
            # if save_traj_dir is not None and (step_idx % save_interval == 0 or step_idx <= 3):
            #     from chroma.data.protein import Protein
            #     # Filter to only chainA (first chain)
            #     first_chain_id = chain_idx[0, 0]
            #     chain_a_mask = (chain_idx[0] == first_chain_id).cpu()
            #     X_current_full = bb_atoms_t.detach().cpu()
            #     X_current_chainA = X_current_full[:, chain_a_mask, :, :]
            #     C_current_chainA = chain_idx[:, chain_a_mask].cpu() + 25
            #     S_current_chainA = torch.ones_like(C_current_chainA)
            #     p_current = Protein.from_XCS(X_current_chainA, C_current_chainA, S_current_chainA)
            #     current_pdb_path = os.path.join(save_traj_dir, f'current_state_step_{step_idx:04d}_chainA.pdb')
            #     p_current.to_PDB(current_pdb_path)
            #     if step_idx <= 3:
            #         print(f'[Motif+Sym] Saved current chainA at step {step_idx}')

            # Model prediction (on full symmetric structure)
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch, recycle=1)

            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append((pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu()))

            # Apply symmetry to model predictions (BEFORE denoising step)
            bb_atoms_pred = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            sym_bb_atoms_pred, _ = self.symmetry.apply_symmetry(
                bb_atoms_pred.squeeze(0).cpu(), S_expanded
            )
            sym_bb_atoms_pred = sym_bb_atoms_pred.to(self._device)
            pred_rotmats_1, pred_trans_1, _ = self.frame_builder.inverse(
                sym_bb_atoms_pred.unsqueeze(0), chain_idx
            )

            # # Save predicted structure after symmetry (only chainA, every 10 steps or first 3 steps)
            # if save_traj_dir is not None and (step_idx % save_interval == 0 or step_idx <= 3):
            #     from chroma.data.protein import Protein
            #     # Filter to only chainA (first chain)
            #     first_chain_id = chain_idx[0, 0]
            #     chain_a_mask = (chain_idx[0] == first_chain_id).cpu()
            #     X_pred_full = sym_bb_atoms_pred.unsqueeze(0).detach().cpu()
            #     X_pred_chainA = X_pred_full[:, chain_a_mask, :, :]
            #     C_pred_chainA = chain_idx[:, chain_a_mask].cpu() + 25
            #     S_pred_chainA = torch.ones_like(C_pred_chainA)
            #     p_pred = Protein.from_XCS(X_pred_chainA, C_pred_chainA, S_pred_chainA)
            #     pred_pdb_path = os.path.join(save_traj_dir, f'predicted_step_{step_idx:04d}_chainA.pdb')
            #     p_pred.to_PDB(pred_pdb_path)
            #     if step_idx <= 3:
            #         print(f'[Motif+Sym] Saved predicted chainA at step {step_idx}')

            # Align each chain's prediction to its motif before denoising step
            if step_idx <= 3:
                print(f'[Motif+Sym] Step {step_idx}: Aligning predictions to motifs per chain...')
            pred_trans_1, pred_rotmats_1 = self._align_prediction_to_motif_per_chain(
                pred_trans_1, pred_rotmats_1, trans_t_1, rotmats_t_1, chain_idx, fixed_mask
            )

            # # Save aligned prediction (every 10 steps or first 3 steps)
            # if save_traj_dir is not None and (step_idx % save_interval == 0 or step_idx <= 3):
            #     from chroma.data.protein import Protein
            #     bb_atoms_aligned = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
            #     X_aligned = bb_atoms_aligned.detach().cpu()
            #     C_aligned = chain_idx + 25
            #     S_aligned = torch.ones_like(C_aligned)
            #     p_aligned = Protein.from_XCS(X_aligned, C_aligned, S_aligned)
            #     aligned_pdb_path = os.path.join(save_traj_dir, f'aligned_step_{step_idx:04d}.pdb')
            #     p_aligned.to_PDB(aligned_pdb_path)
            #     if step_idx <= 3:
            #         print(f'[Motif+Sym] Saved aligned structure at step {step_idx}')

            # Take denoising step (with symmetry-corrected and motif-aligned predictions)
            d_t = t_2 - t_1

            if sampling_method == 'heun':
                trans_t_2 = self.heun_step_R3(d_t, t_1, pred_trans_1, trans_t_1)
                rotmats_t_2 = self.heun_step_SO3(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            else:
                trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
                rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)

            # Re-apply motif constraints to ALL motif positions (via fixed_mask)
            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_m_full, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_m_full, diffuse_mask)

            # Re-apply symmetry constraints
            bb_atoms_pred = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            sym_bb_atoms_pred, _ = self.symmetry.apply_symmetry(
                bb_atoms_pred.squeeze(0).cpu(), S_expanded
            )

            sym_bb_atoms_pred = sym_bb_atoms_pred.to(self._device)

            # Convert back to frames
            rotmats_t_2, trans_t_2, _ = self.frame_builder.inverse(
                sym_bb_atoms_pred.unsqueeze(0), chain_idx
            )

            # Re-apply motif constraints AGAIN after symmetry (critical!)
            # Symmetry overwrites all subunits, so we must restore ALL motif positions
            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_m_full, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_m_full, diffuse_mask)

            # Center based on motif region (keep motif at center)
            # motif_center_current = trans_t_2[:, motif_mask_bool].mean(1, keepdim=True)
            # trans_t_2 = trans_t_2 - motif_center_current

            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            prot_traj.append((trans_t_2, rotmats_t_2))

            # Save intermediate trajectory
            if save_traj_dir is not None and step_idx % save_interval == 0:
                from chroma.data.protein import Protein
                bb_atoms_step = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)

                # Motif should already be correct due to constraint application
                X_step = bb_atoms_step.detach().cpu()
                C_step = chain_idx + 25
                S_step = torch.ones_like(C_step)

                p_step = Protein.from_XCS(X_step, C_step, S_step)
                step_pdb_path = os.path.join(save_traj_dir, f'step_{step_idx:04d}.pdb')
                p_step.to_PDB(step_pdb_path)

            t_1 = t_2

        # Final prediction
        print('[Motif+Sym] Final prediction...')
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1

        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        # Apply motif constraints to ALL motif positions
        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_m_full, diffuse_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_m_full, diffuse_mask)

        # Apply final symmetry
        bb_atoms_final = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)
        sym_bb_atoms_final, sym_S_final = self.symmetry.apply_symmetry(
            bb_atoms_final.squeeze(0).cpu(), S_expanded
        )

        # Convert back to frames and re-apply motif constraints
        sym_bb_atoms_final = sym_bb_atoms_final.to(self._device)
        pred_rotmats_1, pred_trans_1, _ = self.frame_builder.inverse(
            sym_bb_atoms_final.unsqueeze(0), chain_idx
        )
        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_m_full, diffuse_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_m_full, diffuse_mask)

        # Center based on motif region (keep motif at center)
        motif_center_final = pred_trans_1[:, motif_mask_bool].mean(1, keepdim=True)
        pred_trans_1 = pred_trans_1 - motif_center_final

        # Rebuild final structure with motif constraints applied
        bb_atoms_final = self.frame_builder(pred_rotmats_1.float(), pred_trans_1, chain_idx)

        # Prepare output
        X = bb_atoms_final.detach().cpu()
        C = chain_idx + 25
        S = torch.ones_like(C)

        print(f'[Motif+Sym] Generated structure: {X.shape[1]} residues ({asym_len} × {self.symmetry.order})')

        return X, C, S

    def hybrid_binder_sample(
            self,
            model,
            batch,
            num_steps=None,


    ):
        model.eval()
        num_batch = batch['fixed_mask'].shape[0]
        chain_idx = batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        if num_steps is not None:
            self._sample_cfg.num_timesteps = num_steps
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch, recycle=1, is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']

            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            # #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # # 获取当前使用的全局seed
            # current_seed = get_global_seed()
            # seed_prefix = f"{current_seed}_" if current_seed is not None else ""
            #
            # saved_path = f"{seed_prefix}{t.detach().cpu().item()}_binder_sample.pdb"
            # p.to_PDB(saved_path)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        # return [atom37_traj],None

        X = atom37_traj.detach().cpu()
        C = chain_idx.detach().cpu()
        S = model_out['SEQ'].detach().cpu()
        bf = model_out['pred_bf'] * 20

        return X, C, S, bf

    def hybrid_DYN_binder_sample(
            self,
            model,
            batch,
            num_steps=None,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        if num_steps is not None:
            self._sample_cfg.num_timesteps = num_steps
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in tqdm(ts[1:] ,leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1

            tx1=trans_t_1[0]
            tx2 = trans_t_1[1]
            tx0=tx1-tx2

            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']

            pred_trans_1_average = maskwise_avg_replace(model_out['pred_trans'], batch['fixed_mask'])
            x1=model_out['pred_trans'][0]
            x2 = model_out['pred_trans'][1]
            x0=x1-x2

            pred_rotmats_1 = model_out['pred_rotmats']

            pred_rotmats_1_average=parallel_karcher_mean(model_out['pred_rotmats'], batch['fixed_mask'])



            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)


            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            # #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # # 获取当前使用的全局seed
            # current_seed = get_global_seed()
            # seed_prefix = f"{current_seed}_" if current_seed is not None else ""
            #
            # saved_path = f"{seed_prefix}{t.detach().cpu().item()}_binder_sample.pdb"
            # p.to_PDB(saved_path)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        pred_trans_1_average = maskwise_avg_replace(model_out['pred_trans'], batch['fixed_mask'])
        pred_rotmats_1_average = parallel_karcher_mean(model_out['pred_rotmats'], batch['fixed_mask'])

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        # return [atom37_traj],None


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=model_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf

    def hybrid_binder_inverse_noise_sample(
            self,
            model,
            batch,
            inverse_noisy_batch,
            num_steps=None,
            num_opt_steps=500,
            opt_lr=1e-3
    ):
        num_batch = batch['fixed_mask'].shape[0]
        chain_idx = batch['chain_idx']



        # --- 阶段1：通过反解噪声获得初始噪声 (NEW)
        optimized_trans_T, optimized_rotmats_T = invert_noise_via_optimization(
            model, inverse_noisy_batch, num_opt_steps=num_opt_steps, lr=opt_lr
        )

        # 使用反解噪声作为采样起点
        trans_0, rotmats_0 = optimized_trans_T, optimized_rotmats_T
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        if num_steps is not None:
            self._sample_cfg.num_timesteps = num_steps
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        for t_2 in tqdm(ts[1:], leave=True, desc='time step', ncols=100, dynamic_ncols=True, position=0):
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t

            with torch.no_grad():
                model_out = model(batch, recycle=1, is_training=True)

            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']

            d_t = t_2 - t_1

            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # Final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X = atom37_traj.detach().cpu()
        C = chain_idx.detach().cpu()
        S = model_out['SEQ'].detach().cpu()
        bf = model_out['pred_bf'] * 20

        return X, C, S, bf

    def hybrid_binder_side_sample(
            self,
            model,
            batch,
            sidechain,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in ts[1:]:

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=True)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._trans_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            # ###########################
            # atoms4 = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            # X = atoms4.detach().cpu()
            # C = chain_idx.detach().cpu()
            # S = model_out['SEQ'].detach().cpu()
            #
            # bf = model_out['pred_bf'] * 20
            # p = Protein.from_XCSB(X, C, S, bf)
            # saved_path = str(t)+'_.pdb'
            # p.to_PDB(saved_path)

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        print('\n now is designing side chain...')
        # sidechain
        batch['trans_t'] = pred_trans_1
        batch['rotmats_t'] = pred_rotmats_1
        batch['bbatoms_t'] = atom37_traj


        with torch.no_grad():
            sidemodel_out =    sidechain(batch)
        pred_trans_1 = sidemodel_out['pred_trans']
        pred_rotmats_1 = sidemodel_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=sidemodel_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf
    def hybrid_binder_side_sample_inter(
            self,
            model,
            batch,
            sidechain,

    ):
        num_batch=batch['fixed_mask'].shape[0]
        chain_idx=batch['chain_idx']
        trans_0, rotmats_0 = batch['trans_t'], batch['rotmats_t']
        prot_traj = [(trans_0, rotmats_0)]

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]


        for t_2 in tqdm(ts[1:]):

            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch,recycle=1,is_training=False)

            # #update side chain info
            # batch['aatype']=model_out['SEQ']
            # batch['atoms14_b_factors']=model_out['pred_bf']*20
            # batch['chi']=model_out['pred_chi']

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._trans_euler_step(
            #     d_t, t_1, pred_trans_1, trans_t_1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)


            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

            with torch.no_grad():
                batch['trans_t'] = trans_t_2
                batch['rotmats_t'] = rotmats_t_2
                sidemodel_out = sidechain(batch,is_training=True)
            pred_trans_1 = sidemodel_out['pred_trans']
            pred_rotmats_1 = sidemodel_out['pred_rotmats']
            S = sidemodel_out['SEQ'].detach()
            bf = model_out['pred_bf'] * 20
            chi=sidemodel_out['pred_chi']

            batch['chi']=chi
            batch['aatype'] = S
            batch['atoms14_b_factors'] = bf

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch,recycle=1,is_training=False)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=sidemodel_out['SEQ'].detach().cpu()
        bf=model_out['pred_bf']*20

        return X, C, S,bf
    def hybrid_side_sample(
            self,
            model,
            batch

    ):

        chain_idx=batch['chain_idx']

        with torch.no_grad():
            model_out = model(batch,is_training=True)


        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']



        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()


        X=atom37_traj.detach().cpu()
        C=chain_idx.detach().cpu()
        S=model_out['SEQ'].detach().cpu()
        bf=batch['atoms14_b_factors'][...,:4]

        return (X, C, S,bf),model_out



    def hybrid_motif_long_sample(
            self,
            num_batch,
            num_res:list,
            model,
            intervals,
            native_X,
            mode='motif',


    ):


        def make_fixed_mask(mode):

            if mode=='motif':
                # 创建一个全0的tensor
                fixed_maskmask = torch.zeros(num_res, dtype=torch.int)

                # 遍历输入列表，将指定区间和单独位置的值设置为1
                for item in intervals:
                    if isinstance(item, list):  # 如果是区间
                        start, end = item
                        fixed_maskmask[start - 1:end] = 1
                    else:  # 如果是单独的位置
                        fixed_maskmask[item - 1] = 1
            else:
                fixed_maskmask = torch.zeros(num_res, dtype=torch.int)


            return fixed_maskmask.to(self._device)



        res_mask, chain_idx, trans_0, rotmats_0,rg=self._init_complex_backbone(num_batch, num_res)


        C=chain_idx+25
        S=torch.ones_like(C)


        p = Protein.from_XCS(native_X, C, S, )
        p.to_PDB('1a0aA00_motif_native.pdb')


        # print('make fixed mask by interval')
        fixed_mask = make_fixed_mask(mode)
        init_backbone = self.frame_builder(rotmats_0.float(), trans_0, chain_idx)
        init_backbone = init_backbone * (1 - fixed_mask[..., None, None]) + native_X * fixed_mask[..., None, None]

        p = Protein.from_XCS(init_backbone, C, S, )
        p.to_PDB('1a0aA00_motif_init_backbone.pdb')

        # centrelise
        batch={'res_mask':res_mask,'bbatoms':init_backbone,}
        native_X=self._center(batch)['bbatoms']


        p = Protein.from_XCS(native_X, C, S, )
        p.to_PDB('1a0aA00_motif_centrelise.pdb')


        # get new trans0
        rotmats_f,trans_f, _=self.frame_builder.inverse(native_X, chain_idx)


        # update with fixed mask
        # trans_0 = trans_0*(1-fixed_mask[...,None])+trans_f*fixed_mask[...,None]
        # rotmats_0 = rotmats_0*(1-fixed_mask[...,None,None])+rotmats_f*fixed_mask[...,None,None]

        batch = {
            'res_mask': res_mask,
            'chain_idx': chain_idx,}

        ####### motif part##############
        batch['fixed_mask'] = fixed_mask
        # batch['fixed_mask'] = None
        # batch['trans_1'] = trans_f / 10.0
        # batch['rotmats_1'] = rotmats_f

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_f, rotmats_f)]
        clean_traj = []
        for t_2 in tqdm(ts[1:]):

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]


            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            bb_atoms_t = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)
            #batch['bbatoms_t'] = bb_atoms_t*(1-fixed_mask[...,None,None])+native_X*fixed_mask[...,None,None]
            batch['bbatoms_t'] =bb_atoms_t

            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)  #

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']


            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )


            # Take reverse step
            d_t = t_2 - t_1

            trans_t_2 = self._X_Temp_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            # trans_t_2 = self._X_Temp_motif_euler_step(
            #     d_t, t_1,trans_f, pred_trans_1, trans_t_1,fixed_mask,0.1)

            def _center(_X):
                _X = _X - _X.mean(1, keepdim=True)
                return _X

            # trans_t_2 = _center(trans_t_2)
            if self._cfg.self_condition:
                batch['trans_sc'] = trans_t_2

            # rotmats_t_2 = self._rots_motif_euler_step(
            #     d_t, t_1,rotmats_f, pred_rotmats_1, rotmats_t_1,fixed_mask,0.1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)



            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2


            #######test motif fixed area
            bb_atoms_tss = self.frame_builder(rotmats_t_2.float(), trans_t_2, chain_idx)
            fixed_area_n = trans_f.cpu() * fixed_mask[..., None].cpu()
            fixed_area_p = trans_t_2.cpu() * fixed_mask[..., None].cpu()

            # p = Protein.from_XCS(bb_atoms_tss, C, S, )
            # p.to_PDB('1a0aA00_motif_'+str(t_2)+'.pdb')

            RMSD = torch.sum((fixed_area_n - fixed_area_p) ** 2, dim=-1)
            RMSD = torch.sqrt(RMSD).mean()
            print('eluer out: ',RMSD)


        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['bbatoms_t'] = self.frame_builder(rotmats_t_1.float(), trans_t_1, chain_idx)

        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        # pred_trans_1 = _center(pred_trans_1)
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        # atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        atom37_traj = self.frame_builder(pred_rotmats_1, pred_trans_1, chain_idx).detach().cpu()

        X=atom37_traj
        C=chain_idx+25
        S=torch.ones_like(C)


        fixed_area_n=trans_f.cpu()*fixed_mask[...,None].cpu()
        fixed_area_p = pred_trans_1.cpu() * fixed_mask[..., None].cpu()

        RMSD=torch.sum((fixed_area_n-fixed_area_p)**2,dim=-1)
        RMSD=torch.sqrt(RMSD.mean(), )
        print(RMSD)
        return X,C,S





def _debug_viz_gradients(
    pml_file, X_list, dX_list, C, S, arrow_length=2.0, name="gradient", color="red"
):
    """ """
    lines = [
        "from pymol.cgo import *",
        "from pymol import cmd",
        f'color_1 = list(pymol.cmd.get_color_tuple("{color}"))',
        'color_2 = list(pymol.cmd.get_color_tuple("blue"))',
    ]

    with open(pml_file, "w") as f:
        for model_ix, X in enumerate(X_list):
            print(model_ix)
            lines = lines + ["obj_1 = []"]

            dX = dX_list[model_ix]
            scale = dX.norm(dim=-1).mean().item()
            X_i = X
            X_j = X + arrow_length * dX / scale

            for a_ix in range(4):
                for i in range(X.size(1)):
                    x_i = X_i[0, i, a_ix, :].tolist()
                    x_j = X_j[0, i, a_ix, :].tolist()
                    lines = lines + [
                        f"obj_1 = obj_1 + [CYLINDER] + {x_i} + {x_j} + [0.15]"
                        " + color_1 + color_1"
                    ]
            lines = lines + [f'cmd.load_cgo(obj_1, "{name}", {model_ix+1})']
            f.write("\n" + "\n".join(lines))
            lines = []


def _debug_viz_XZC(X, Z, C, rgb=True):
    from matplotlib import pyplot as plt

    if len(X.shape) > 3:
        X = X.reshape(X.shape[0], -1, 3)
    if len(Z.shape) > 3:
        Z = Z.reshape(Z.shape[0], -1, 3)
    if C.shape[1] != X.shape[1]:
        C_expand = C.unsqueeze(-1).expand(-1, -1, 4)
        C = C_expand.reshape(C.shape[0], -1)

    # C_mask = expand_chain_map(torch.abs(C))
    # X_expand = torch.einsum('nix,nic->nicx', X, C_mask)
    # plt.plot(X_expand[0,:,:,0].data.numpy())
    N = X.shape[1]
    Ymax = torch.max(X[0, :, 0]).item()
    plt.figure(figsize=[12, 4])
    plt.subplot(2, 1, 1)

    plt.bar(
        np.arange(0, N),
        (C[0, :].data.numpy() < 0) * Ymax,
        width=1.0,
        edgecolor=None,
        color="lightgrey",
    )
    if rgb:
        plt.plot(X[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(X[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(X[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "r", linewidth=0.5)
        plt.plot(Z[0, :, 1].data.numpy(), "g", linewidth=0.5)
        plt.plot(Z[0, :, 2].data.numpy(), "b", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("RInverse @ [X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    else:
        plt.plot(X[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.xlim([0, N])
        plt.grid()
        plt.title("X")
        plt.xticks([])
        plt.subplot(2, 1, 2)
        plt.plot(Z[0, :, 0].data.numpy(), "k", linewidth=0.5)
        plt.plot(C[0, :].data.numpy(), "orange")
        plt.xlim([0, N])
        plt.grid()
        plt.title("Inverse[X]")
        plt.xticks([])
        plt.savefig("xzc.pdf")
    exit()


