from scipy.spatial.distance import pdist
import numpy as np
import torch

"""metrics"""


def compute_all_metrics(pred, gt, conf):  # 修改：少了一个gt_multi，多了一个conf
    """
    评测各个指标。因为和HumanMAC的指标有很大不同，所以一整个重写了
    :param pred: 模型的预测结果，[K, t_pred, 2 * joint_num]
    :param gt: ground truth，[t_pred, 2 * joint_num]
    :param conf: 传感器的conf值，[t_pred, joint_num]
    :param global_max: 最大值，用于把归一化的给乘回去

    :return apd, ade, fde: 三个指标
    """
    # APD不动
    if pred.shape[0] == 1:
        diversity = 0.0
    else: # 修改：源代码这里少了else
        pred_tensor = torch.from_numpy(pred)  # 转换为torch的tensor，用numpy算这个会很复杂
        dist_diverse = torch.pdist(pred_tensor.reshape(pred_tensor.shape[0], -1))
        diversity = dist_diverse.mean().item()

    # 把格式变回去，不然算起来不方便
    K, T_pred, _ = pred.shape
    J = conf.shape[1]
    joint_dim = 2  # 2D
    tau = 0.1

    pred = pred.reshape(K, T_pred, J, joint_dim) # [K, t_pred, joint_num, 2]
    gt = gt.reshape(T_pred, J, joint_dim)  # [t_pred, joint_num, 2]

    de = np.linalg.norm(pred - gt[None, ...], axis=-1)  #[K, t_pred, joint_num]

    ade_k = []
    fde_k = []

    for k in range(K):
        ade_j = []
        fde_j = []

        for j in range(J):
            # Mask for valid frames for joint j
            valid_mask = conf[:, j] >= tau
            valid_indices = np.where(valid_mask)[0]

            # ADE
            if len(valid_indices) > 0:
                ade = np.mean(de[k, valid_indices, j])
                ade_j.append(ade)

                # FDE: use last valid frame
                t_star = valid_indices[-1]
                fde = de[k, t_star, j]
                fde_j.append(fde)

        if len(ade_j) > 0:
            ade_k.append(np.mean(ade_j))
        if len(fde_j) > 0:
            fde_k.append(np.mean(fde_j))

    final_ade = np.min(ade_k) if ade_k else float('inf')
    final_fde = np.min(fde_k) if fde_k else float('inf')


    return diversity, final_ade, final_fde  # 修改：取消多模态