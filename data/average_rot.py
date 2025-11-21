import torch
import torch.nn.functional as F

def karcher_mean_quaternion_torch(quats: torch.Tensor, max_iter=30, tol=1e-6):
    """
    quats: [K, 4], unit quaternions
    return: [4] mean quaternion
    """
    quats = F.normalize(quats, dim=-1)

    # 对齐方向
    ref = quats[0]
    dots = (quats * ref).sum(dim=-1)
    quats = torch.where(dots.unsqueeze(-1) < 0, -quats, quats)

    def log_map(q, mu):
        dot = (q * mu).sum(-1, keepdim=True).clamp(-1.0, 1.0)
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

    mu = quats[0].clone().unsqueeze(0)  # [1, 4]
    for _ in range(max_iter):
        log_vecs = log_map(quats, mu)  # [K, 4]
        delta = log_vecs.mean(dim=0, keepdim=True)  # [1, 4]
        if delta.norm() < tol:
            break
        mu = exp_map(delta, mu)
        mu = F.normalize(mu, dim=-1)

    return mu.squeeze(0)  # [4]

def quaternion_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: [4] or [B, 4] unit quaternion (x, y, z, w)
    return: [3, 3] or [B, 3, 3]
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R[0] if R.shape[0] == 1 else R

def rotmat_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    R: [3, 3] or [B, 3, 3]
    return: [4] or [B, 4] unit quaternion (x, y, z, w)
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    for i in range(B):
        t = trace[i]
        if t > 0.0:
            s = torch.sqrt(t + 1.0) * 2
            q[i, 3] = 0.25 * s
            q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
            q[i, 1] = (R[i, 0, 2] - R[i, 2, 0]) / s
            q[i, 2] = (R[i, 1, 0] - R[i, 0, 1]) / s
        else:
            if R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
                s = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
                q[i, 3] = (R[i, 2, 1] - R[i, 1, 2]) / s
                q[i, 0] = 0.25 * s
                q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
                q[i, 2] = (R[i, 0, 2] + R[i, 2, 0]) / s
            elif R[i, 1, 1] > R[i, 2, 2]:
                s = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
                q[i, 3] = (R[i, 0, 2] - R[i, 2, 0]) / s
                q[i, 0] = (R[i, 0, 1] + R[i, 1, 0]) / s
                q[i, 1] = 0.25 * s
                q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s
            else:
                s = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
                q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / s
                q[i, 0] = (R[i, 0, 2] + R[i, 2, 0]) / s
                q[i, 1] = (R[i, 1, 2] + R[i, 2, 1]) / s
                q[i, 2] = 0.25 * s
    return F.normalize(q, dim=-1) if B > 1 else F.normalize(q[0], dim=0)

def batch_masked_karcher_mean(rot_mats: torch.Tensor, fixed_mask: torch.Tensor) -> torch.Tensor:
    """
    rot_mats: [B, N, 3, 3]
    fixed_mask: [B, N], 0 = to replace with Karcher mean
    return: [B, N, 3, 3] with masked positions replaced
    """
    B, N = fixed_mask.shape
    device = rot_mats.device
    rot_out = rot_mats.clone()

    for n in range(N):
        valid_b = (fixed_mask[:, n] == 0).nonzero(as_tuple=False).squeeze(-1)
        if valid_b.numel() == 0:
            continue
        R_subset = rot_mats[valid_b, n]  # [K, 3, 3]
        quats = rotmat_to_quaternion(R_subset)  # [K, 4]
        mean_q = karcher_mean_quaternion_torch(quats)  # [4]
        mean_R = quaternion_to_rotmat(mean_q)         # [3, 3]
        rot_out[valid_b, n] = mean_R
    return rot_out
