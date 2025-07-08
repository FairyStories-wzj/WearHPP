import argparse
import sys

from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *

sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

from data_loader.dataset_xrf2 import Xrf2Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))  # 设备

    parser.add_argument('--cfg', default='h36m', help='h36m or humaneva')  # 数据集名字
    parser.add_argument('--ckpt', type=str, default=None)  # 预读检查点， 为空表示不用
    parser.add_argument('--mode', default='train', help='train / eval / pred / switch/ control/ zero_shot')  # 模式
    parser.add_argument('--iter', type=int, default=0)  # 最大训练轮数，为0表示不限
    parser.add_argument('--seed', type=int, default=43)  # 随机数种子

    parser.add_argument('--save_model_interval', type=int, default=10)  # 每隔几轮保存一次模型
    parser.add_argument('--save_gif_interval', type=int, default=10)  # 每隔几轮画一次动画

    parser.add_argument('--ema', type=bool, default=True)  # 是否使用ema
    parser.add_argument('--vis_switch_num', type=int, default=10)  # 不知道是干什么的
    parser.add_argument('--vis_col', type=int, default=5)  # 画动画的时候画几列
    parser.add_argument('--vis_row', type=int, default=3)  # 画动画的时候画几行

    # 以下是不用的参数
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])  # 在哪个点学习率下降，不用
    parser.add_argument('--gamma', type=float, default=0.9)  # 原版学习率下降速度，不用
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)  # 多模态，不用
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)  # 多模态，不用
    parser.add_argument('--save_metrics_interval', type=int, default=100)  # 多模态，不用

    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    # dataset, dataset_multi_test = dataset_split(cfg)  修改：改成Xrf2的读取
    dataset = {}
    dataset['test'] = Xrf2Dataset(dataset_path=cfg.test_path, t_his=cfg.t_his, t_pred=cfg.t_pred)

    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        dataset['train'] = Xrf2Dataset(dataset_path=cfg.train_path, t_his=cfg.t_his, t_pred=cfg.t_pred)
        # prepare full evaluation dataset
        # multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)  修改：取消多模态
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            # multimodal_dict=multimodal_dict,  修改：取消多模态
            multimodal_dict=None,
            logger=logger,
            tb_logger=tb_logger)
        trainer.loop()

    elif args.mode == 'eval':
        # 修改：改成了适配于Xrf2数据集的
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['net_state_dict'])
        # prepare full evaluation dataset
        # multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        # compute_stats(diffusion, multimodal_dict, model, logger, cfg)
        model_dict = dataset['test'].get_evaluation_samples()
        compute_stats(diffusion, model_dict, model, logger, cfg)

    elif args.mode == 'my_eval':  # 绘制舍弃conf较小的关节的动画
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            # multimodal_dict=multimodal_dict,  修改：取消多模态
            multimodal_dict=None,
            logger=logger,
            tb_logger=tb_logger)
        trainer.draw_animation_without_low_confs()

    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['net_state_dict'])
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
