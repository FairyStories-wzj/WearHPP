import csv

import pandas as pd
from Diffusion_Model_wzj.utils.metrics import *
from tqdm import tqdm
from Diffusion_Model_wzj.utils import *
from Diffusion_Model_wzj.utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, multimodal_dict, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    def get_prediction(data, model_select):
        traj_np = data[..., :, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 2 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    # 为Xrf2制作一个评估数据提取器，返回对象包括
    # 一个字典，包括'gt_group'，'data_group', 'num_samples'这三个键，它们的值分别是：
    # data_group: numpy数组，(samples, frame, joint, 2)，其中samples是采样的总数
    # gt_group: numpy张量，(samples, frame - t_his, (joint - 1) * 2)，data_group中只保留预测的部分、去掉0号关节、最后两维合并
    # conf_group: numpy张量，(samples, frame - t_his, joint - 1)，代表了相应的conf，和gt有位置对应关系
    # num_samples: 样本的总数
    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    # traj_gt_arr = multimodal_dict['traj_gt_arr']  # 修改：取消多模态
    conf_group = multimodal_dict['conf_group']  # 修改：conf_group是新增的
    num_samples = multimodal_dict['num_samples']
    # global_max = multimodal_dict['global_max']

    stats_names = ['APD', 'ADE', 'FDE']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    K = 50  # 目测应该是预测50次求平均，所以如果跑得慢的话这里调整一下
    pred = []
    print("正在出future pose：")

    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5187, 125, 48] in h36m
            pred = pred[:, :, cfg.t_his:, :]
            # Use GPU to accelerate  拿到里面去做
            # try:
            #     gt_group = torch.from_numpy(gt_group).to('cuda')
            # except:
            #     pass
            # try:
            #     pred = torch.from_numpy(pred).to('cuda')
            # except:
            #     pass
            # pred [50, 5187, 100, 48]
            for j in range(0, num_samples):
                # apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                #                                                         gt_group[j][np.newaxis, ...])
                                                                        # traj_gt_arr[j])  # 修改：取消多模态
                apd, ade, fde = compute_all_metrics(pred[:, j, :, :],
                                                    # gt_group[j][np.newaxis, ...],
                                                    gt_group[j],  # 修改：输入格式
                                                    conf_group[j])
                                                    # global_max)

                if not (np.isinf(apd) or np.isinf(ade) or np.isinf(fde) or
                        np.isnan(apd) or np.isnan(ade) or np.isnan(fde)):
                    # Ground truth中所有的关节在预测阶段全部丢完是有可能的，你可以看一下8_livingroom_1.csv的900~1000行的那些数据
                    # 所以需要处理一下inf的问题
                    stats_meter['APD']['HumanMAC'].update(apd)
                    stats_meter['ADE']['HumanMAC'].update(ade)
                    stats_meter['FDE']['HumanMAC'].update(fde)
                else:
                    print(f"Warning: Invalid metric encountered at sample {j}: APD={apd}, ADE={ade}, FDE={fde}")

            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            # new_meter['HumanMAC'] = new_meter['HumanMAC'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)
