"""
新数据集csv文件的格式：
person_scene_action_"pose".csv
"time", "keypoints", \\
time, [[x0, y0, conf0], [x1, y1, conf1], ...], ...] \\
time, [[x0, y0, conf0], [x1, y1, conf1], ...], ...] \\

存到内存里的数据的格式：
data =
{
    "0":  // 人的编号
        {
            "kitchen":  // 场景的名字
            {
                "1":  // 动作序列的编号
                    [
                        [  // 第零帧
                            [x0, y0], [x1, y1], [x2, y2], ...
                        ],
                        [ ... ], // 第一帧
                        ...
                    ],
                "2":
                    [...]
            },
            "livingroom": {...},
            ...
        },
    "1": {...}
}

imu =
{
    "0":  // 人的编号
        {
            "kitchen":  // 场景的名字
            {
                "1":  // 动作序列的编号
                    [
                        [  // 第零帧
                            [imu0的6个数值*], [imu1的6个数值], ...
                        ],
                        [ ... ], // 第一帧
                        ...
                    ],
                "2":
                    [...]
            },
            "livingroom": {...},
            ...
        },
    "1": {...}
}

conf =  // conf和坐标分开放
{
    "0":
        {
            "kitchen":{
                "1":
                    [
                        [  // 第零帧
                            [conf0], [conf1], ...  // 之所以每个都单独开一个列表是为了与data的结构保持一致
                        ],
                        [  // 第一帧
                            [conf0], [conf1], ...
                        ],
                        ...
                    ],
                "2":
                    [...]
            },
            "livingroom": {...},
            ...
        },
    "1": {...}
}
"""
import os
import sys
import ast
import json
import pandas as pd
import random
import torch
import numpy as np
from tqdm import tqdm
from Diffusion_Model_wzj.data_loader.skeleton_xrf2 import Xrf2Skeleton
from IMU_to_pose.inference import infer_pose


def loadToMem(dataset_path):
    """
    把数据集读进来，并组织成上面的形式
    """
    data, conf = {}, {}
    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            print("正在读取：", dataset_path, file)

            # 初始化
            person, scene, action = file.split("_")[:3]
            if not data.get(person):
                data[person], conf[person] = {}, {}
            if not data[person].get(scene):
                data[person][scene], conf[person][scene] = {}, {}
            data[person][scene][action], conf[person][scene][action] = [], []

            df = pd.read_csv(os.path.join(dataset_path, file), encoding="utf-8")
            df['keypoints'] = df['keypoints'].apply(ast.literal_eval)

            for frame in df['keypoints']:
                coords = []
                confs = []

                for x, y, c in frame:
                    coords.append([x, y])
                    confs.append([c])  # 包装成列表以保持结构一致

                data[person][scene][action].append(coords)
                conf[person][scene][action].append(confs)

    # 读取mean和std用以进行逆归一化
    mean, std = {}, {}
    mean['kitchen'], mean['livingroom'], mean['office'] = {}, {}, {}
    std['kitchen'], std['livingroom'], std['office'] = {}, {}, {}
    for scene in mean.keys():
        with open(f'./Diffusion_Model_wzj/data_loader/pose_zscore_param_{scene}.json') as f:
            mean_std = json.load(f)
            mean[scene] = [mean_std['mean'][i:i+3] for i in range(0, len(mean_std['mean']), 3)]  # [joint, 3]
            mean[scene] = [joint[:2] for joint in mean[scene]]
            std[scene] = [mean_std['std'][i:i + 3] for i in range(0, len(mean_std['std']), 3)]  # [joint, 3]
            std[scene] = [joint[:2] for joint in std[scene]]

    # with open("data_test.json", "w") as f:
    #     json.dump(data, f, indent=4)
    # with open("conf_test.json", "w") as f:
    #     json.dump(conf, f, indent=4)
    # return data, conf, global_max
    #
    # with open("mean_test.json", "w") as f:
    #     json.dump(mean, f, indent=4)
    # with open("std_test.json", "w") as f:
    #     json.dump(std, f, indent=4)

    return data, conf, mean, std

# def fault_handling(data_seq, conf_seq):
#     """
#     处理数据瑕疵
#     :param data_seq: 一个形状为[T, J, 2]的列表
#     :param conf_seq: 一个形状为[T, J, 1]的列表
#     :return data_seq: 修改过的data_seq
#     这个函数无法处理那种一整段都丢掉的情况。对于这种情况，我会在外面处理。
#     """
#     joint_num = len(data_seq[0])
#     eps = 0.1
#
#     for joint in range(joint_num):
#         modified = [False] * len(data_seq)  # 标记哪些地方改过了，防止重复更改
#
#         # step1: 边界处理，对于从一开始就丢点的，用它第一次出现的位置；对于最后丢点的，用它最后出现的位置
#         for frame in range(len(data_seq)):
#             if conf_seq[frame][joint][0] >= eps:
#                 for pre_frame in range(frame):
#                     if not modified[pre_frame]:
#                         data_seq[pre_frame][joint] = data_seq[frame][joint][:]  # 在后面加一个[:]可以防止被共享
#                         modified[pre_frame] = True
#                 break
#
#         for frame in range(len(data_seq) - 1, -1, -1):
#             if conf_seq[frame][joint][0] >= eps:
#                 for nxt_frame in range(frame + 1, len(data_seq)):
#                     if not modified[nxt_frame]:
#                         data_seq[nxt_frame][joint] = data_seq[frame][joint][:]
#                         modified[nxt_frame] = True
#                 break
#
#         # print(modified)
#
#         # step2: 正常处理，对于一段丢点，平分前一次出现和后一次出现
#         lst_appear, nxt_appear = -1, -1  # 前一次出现和后一次出现的位置
#         for frame in range(len(data_seq)):
#             if not modified[frame] and conf_seq[frame][joint][0] < eps and lst_appear == -1:  # 找到第一个丢点的
#                 lst_appear = frame - 1
#             if conf_seq[frame][joint][0] >= eps and lst_appear != -1 and nxt_appear == -1:  # 找到后一次出现的位置
#                 nxt_appear = frame
#
#                 step_x = (data_seq[nxt_appear][joint][0] - data_seq[lst_appear][joint][0]) / (nxt_appear - lst_appear)  # 平均分配
#                 step_y = (data_seq[nxt_appear][joint][1] - data_seq[lst_appear][joint][1]) / (nxt_appear - lst_appear)
#                 x0, y0 = data_seq[lst_appear][joint]
#
#                 for lost_frame in range(lst_appear + 1, nxt_appear):
#                     data_seq[lost_frame][joint] = [x0 + step_x * (lost_frame - lst_appear),
#                                                    y0 + step_y * (lost_frame - lst_appear)]
#
#                 lst_appear, nxt_appear = -1, -1
#
#     return data_seq


class Xrf2Dataset:
    def __init__(self, t_his=15 * 25, t_pred=15 * 5, dataset_path=None):
        """
        t_his：做历史的有多少帧
        t_pred：做预测的有多少帧
        dataset_path：数据集放在哪里，路径的最后一个文件夹应该是train或者test
        """
        self.data, self.conf, self.mean, self.std = loadToMem(dataset_path)
        self.t_his, self.t_pred = t_his, t_pred

    # def getGlobalMax(self):
    #     # return self.global_max
    #     return 1

    def generator_train(self, batch_size=8, num_samples=1000):
        """
        一个迭代器
        采样，生成一个(batch_size, frame_num, joint_num, 2)的张量，为对应gt
        返回一个(batch_size, 4)的列表，代表每个batch的person, scene, action, frame_start, frame_end
        返回一个(batch_size, frame_num, joint_num, 1)的张量，为对应的conf
        :param num_samples: 总共需要采多少样
        会处理数据为0的情况
        """
        samples_generated = 0
        pbar = tqdm(total=num_samples)
        while samples_generated + batch_size <= num_samples:  # 最后一组不够一个batch就不要了
            data, label, conf = [], [], []
            for _ in range(batch_size):
                # 选取人、动作序列和开始帧
                subject = random.choice(list(self.data.keys()))
                scene = random.choice(list(self.data[subject].keys()))
                action = random.choice(list(self.data[subject][scene].keys()))

                # 抽取数据
                t_total = self.t_his + self.t_pred
                while True:
                    frame_start = random.randint(0, len(self.data[subject][scene][action]) - t_total)
                    data_seq = self.data[subject][scene][action][frame_start:frame_start + t_total]  # [T, J, 2]
                    conf_seq = self.conf[subject][scene][action][frame_start:frame_start + t_total]  # [T, J, 1]

                    # 检查有没有一整个序列都丢掉的关节
                    conf_seq_np = np.array(conf_seq).squeeze(-1)
                    if not np.any(np.all(conf_seq_np < 0.1, axis=0)):
                        break
                    # print("警告：存在一整个序列都不可信的关节，重新采样")

                # data_seq = fault_handling(data_seq, conf_seq)

                data_tensor = torch.tensor(data_seq, dtype=torch.float32)
                conf_tensor = torch.tensor(conf_seq, dtype=torch.float32)

                # data_tensor = data_tensor / self.global_max

                data.append(data_tensor)
                conf.append(conf_tensor)
                label.append({
                    'person': subject,
                    'scene': scene,
                    'action': action,
                    'frame_start': frame_start,
                    'frame_end': frame_start + t_total - 1,
                })

            yield torch.stack(data), label, torch.stack(conf)
            samples_generated += batch_size

            pbar.update(batch_size)
        pbar.close()

    def sample(self, train_path_imu=None, test_path_imu=None, ckpt_imu=None):
        """
        画图用的，随机取一个gt，返回格式(1, frame, joint, 2)
        返回的是numpy数组，不是张量

        返回四个东西，一个是gt，格式(frame, joint, 2)；
        一个是imu产生的对应时间段的pose，格式(frame, joint, 2)
        然后是两个用于逆归一化的mean和std，格式(joint, 2)

        """
        # TODO：只做了test没做train
        subject = random.choice(list(self.data.keys()))
        scene = random.choice(list(self.data[subject].keys()))
        action = random.choice(list(self.data[subject][scene].keys()))

        # 抽取gt
        t_total = self.t_his + self.t_pred
        frame_start = random.randint(0, len(self.data[subject][scene][action]) - t_total)
        data_seq = self.data[subject][scene][action][frame_start:frame_start + t_total]
        # conf_seq = self.conf[subject][scene][action][frame_start:frame_start + t_total]
        # data_seq = fault_handling(data_seq, conf_seq)
        # data_array = np.array(data_seq) / self.global_max
        gt = np.array(data_seq)

        # 计算imu
        # 先找是在训练集还是测试集里
        if os.path.exists(os.path.join(test_path_imu, f'{subject}_{scene}_{action}_imu.csv')):
            imu_path = test_path_imu + f'\\{subject}_{scene}_{action}_imu.csv'
        else:
            imu_path = train_path_imu + f'\\{subject}_{scene}_{action}_imu.csv'
        pred_pose = infer_pose(
            model_pth=ckpt_imu,
            imu_csv=imu_path,
            start=frame_start,
            end=frame_start + t_total - 1,  # 闭区间
            mask_probs=None,
            device='cuda',
            chunk_size=256
        )

        return gt, pred_pose, self.mean[scene], self.std[scene]


    # def sample_conf(self):
    #     """
    #     和sample函数的功能完全相同，但是这个多一个返回值：需要舍弃的点，用来绘制“去掉conf低的点的动画”
    #     :return joint_deleted: 一个长度为2的列表，包含需要舍弃的关节
    #     """
    #     subject = random.choice(list(self.data.keys()))
    #     scene = random.choice(list(self.data[subject].keys()))
    #     action = random.choice(list(self.data[subject][scene].keys()))
    #
    #     # 抽取数据
    #     t_total = self.t_his + self.t_pred
    #     frame_start = random.randint(0, len(self.data[subject][scene][action]) - t_total)
    #     data_seq = self.data[subject][scene][action][frame_start:frame_start + t_total]
    #     conf_seq = self.conf[subject][scene][action][frame_start:frame_start + t_total]
    #
    #     data_seq = fault_handling(data_seq, conf_seq)
    #
    #     # data_array = np.array(data_seq) / self.global_max
    #     data_array = np.array(data_seq)
    #
    #     # 求取conf平均值
    #     joint_num = len(self.conf[subject][scene][action][0])
    #     confs = np.zeros(joint_num)
    #     for frame in range(frame_start, frame_start + t_total):
    #         confs += [self.conf[subject][scene][action][frame][j][0] for j in range(joint_num)]
    #     joint_deleted = np.argsort(confs)[:2]
    #
    #     return data_array[None, ...], joint_deleted

    @property  # 用这个可以让函数返回实例，而不是另一个函数
    def skeleton(self):
        return Xrf2Skeleton()

    def get_evaluation_samples(self, step=25, test_path_imu=None, ckpt_imu=None):
        """
        产生用于评估的采样集
        """
        samples = {
            'data_group': [],
            'gt_group': [],
            'conf_group': [],
            'num_samples': 0,
        }

        # conf_seqs = []  # 后面数据优化用

        for person in self.data.keys():
            for scene in self.data[person].keys():
                for action in self.data[person][scene].keys():
                    t_total = self.t_his + self.t_pred
                    for frame_start in range(0, len(self.data[person][scene][action]) - t_total, step):
                        samples['num_samples'] += 1

                        data_seq = self.data[person][scene][action][frame_start:frame_start + t_total]
                        data_array = np.array(data_seq)
                        samples['data_group'].append(data_array)

                        conf_seq = self.conf[person][scene][action][frame_start:frame_start + t_total]  # (frames, joints, 1)
                        # conf_seqs.append(conf_seq)  # (samples, frames, joints, 1)
                        pred_conf_seq = np.array(conf_seq)[self.t_his:, :, :].squeeze(-1)  # (t_pred, joints)
                        samples['conf_group'].append(pred_conf_seq)

        samples['data_group'] = np.stack(samples['data_group'], axis=0)

        all_data = samples['data_group'][..., :, :].reshape(samples['data_group'].shape[0], samples['data_group'].shape[1], -1)
        samples['gt_group'] = all_data[:, self.t_his:, :].copy()
        # ground truth不能动它，因此不做数据优化

        # samples['data_group'] /= self.global_max  # 把归一化挪到这里，这样gt就不用乘回去了

        # data_group是用来给模型做输入的，所以需要做数据优化
        # for sample in range(len(samples['data_group'])):
        #     data_seq = samples['data_group'][sample].tolist()
        #     conf_seq = conf_seqs[sample]
        #     data_seq = fault_handling(data_seq, conf_seq)
        #     samples['data_group'][sample] = np.array(data_seq)

        samples['conf_group'] = np.stack(samples['conf_group'], axis=0)

        # 把data_group用imu重新读一遍，然后计算pose
        samples['data_group'] = []
        for person in self.data.keys():
            for scene in self.data[person].keys():
                for action in self.data[person][scene].keys():
                    t_total = self.t_his + self.t_pred
                    for frame_start in range(0, len(self.data[person][scene][action]) - t_total, step):
                        print(f"正在imu出pose：{person}, {scene}, {action}, {frame_start}")
                        pred_pose = infer_pose(
                            model_pth=ckpt_imu,
                            imu_csv=test_path_imu + f'\\{person}_{scene}_{action}_imu.csv',
                            start=frame_start,
                            end=frame_start + t_total - 1,  # 闭区间
                            mask_probs=None,
                            device='cuda',
                            chunk_size=256
                        )
                        samples['data_group'].append(pred_pose)

        samples['data_group'] = np.stack(samples['data_group'], axis=0)
        print(samples['data_group'].shape)

        return samples


# xrf2_dataset = Xrf2Dataset(dataset_path="E:\\Xrf2\\test")
# sample = xrf2_dataset.get_evaluation_samples()
# conf_group = sample['conf_group']
# print(conf_group.shape)
# print(conf_group[0])
# gt_group = sample['gt_group']
# print(gt_group.shape)
# print(gt_group[0])
# data_group = sample['data_group']
# print(data_group[0])
# print(data_group.shape)
# print(conf_group.shape, sample['num_samples'])
# plt.hist(conf_group.flatten(), bins=50)  # flatten()展开成一维
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Data distribution histogram')
# plt.show()

# print(xrf2_dataset.sample().shape)
# for data_sample, conf_sample in xrf2_dataset.generator_train():
#     print(data_sample)

# test_data_seq = [
#     [
#         [0, 0],
#         [0, 0],
#         [2, 2]
#     ],
#     [
#         [0, 0],
#         [0, 0],
#         [0, 0]
#     ],
#     [
#         [0, 0],
#         [0, 0],
#         [0, 0]
#     ],
#     [
#         [8, 8],
#         [4, 4],
#         [0, 0]
#     ]
# ]
#
# test_conf_seq = [
#     [
#         [0],
#         [0],
#         [1]
#     ],
#     [
#         [0],
#         [0],
#         [0]
#     ],
#     [
#         [0],
#         [0],
#         [0]
#     ],
#     [
#         [1],
#         [1],
#         [0]
#     ]
# ]
#
# print(fault_handling(test_data_seq, test_conf_seq))