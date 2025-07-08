import copy
import time

from torch import optim, nn

from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
                 dataset,
                 cfg,
                 multimodal_dict,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0

        self.lrs = []
        self.lr = 0

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

    def loop(self):
        self.before_train()
        # for self.iter in range(1, self.cfg.num_epoch):
        self.iter += 1  # 从上一次的检查点开始接着往下，所以要加1
        while self.iter <= self.cfg.num_epoch:
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

            self.iter += 1

    def before_train(self):
        # 检查点中断重读机制
        # 修改：把学习率下降方式改成了每10轮下降一次
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        if self.cfg.ckpt and os.path.exists(self.cfg.ckpt):
            print('=> loading checkpoint from', self.cfg.ckpt)
            checkpoint = torch.load(self.cfg.ckpt)
            self.model.load_state_dict(checkpoint['net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.iter = checkpoint['iter']
            self.lr = checkpoint['lr']
        else:
            self.lr = self.cfg.lr
            print("Start a new training...")

        # self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
        #                                                    gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()

        # 修改：使用了自己写的Xrf2的数据提取器
        # self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
        #                                                                 batch_size=self.cfg.batch_size)
        self.generator_train = self.dataset['train'].generator_train(num_samples=self.cfg.num_data_sample,
                                                                     batch_size=self.cfg.batch_size)

        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for traj_np, traj_conf in self.generator_train:  # 修改：多一个conf
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 2) -> (N, t_his + t_pre, 2 * joints - 1)
                # discard the root joint and combine xy coordinate
                traj_np = traj_np[..., :, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

            # train
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(traj_dct, t)
            predicted_noise = self.model(x_t, t, mod=traj_dct_mod)

            # 损失函数一：DCT后的全时间序列
            loss1 = self.criterion(predicted_noise, noise)

            # 损失函数二：iDCT后的未来部分
            predicted_noise_est = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], predicted_noise)  # (batch, frame, joint * 2)
            noise_est = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], noise)

            predicted_noise_est = predicted_noise_est[:, self.cfg.t_his:, :]
            noise_est = noise_est[:, self.cfg.t_his:, :]

            loss2 = self.criterion(predicted_noise_est, noise_est)

            # 最终的损失函数是两者的加权平均，再乘上conf的平均值
            alpha = 0  # 超参数：加权平均系数
            loss_total = (alpha * loss1 + (1 - alpha) * loss2) * traj_conf.mean()

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss_total.item())
            self.tb_logger.add_scalar('Loss/train', loss_total.item(), self.iter)

            del loss_total, loss1, loss2, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_train_step(self):
        # self.lr_scheduler.step()
        # self.lrs.append(self.optimizer.param_groups[0]['lr'])

        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            # self.lrs[-1]))
                                                                            self.lr))
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.dataset['train'], self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))
                             # global_max=self.dataset['train'].getGlobalMax())

    def draw_animation_without_low_confs(self):
        """
        绘制舍弃conf较小的关节的动画
        """
        if self.cfg.ckpt and os.path.exists(self.cfg.ckpt):
            print('=> loading checkpoint from', self.cfg.ckpt)
            checkpoint = torch.load(self.cfg.ckpt)
            self.model.load_state_dict(checkpoint['net_state_dict'])

        pose_gen = pose_generator(self.dataset['test'], self.model, self.diffusion, self.cfg, mode='gif_without_low_confs')
        render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                         output=os.path.join(self.cfg.gif_dir, f'animation_without_low_confs.gif'))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        # 测试集的修改类似于训练集
        # self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
        #                                                              batch_size=self.cfg.batch_size)
        self.generator_val = self.dataset['test'].generator_train(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np, traj_conf in self.generator_val:  # 修改：增加了conf
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 2) -> (N, t_his + t_pre, 2 * joints)
                # discard the root joint and combine xy coordinate
                traj_np = traj_np[..., :, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad,
                                        self.cfg.zero_index)  #
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 2 * (joints - 1)]

                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
                loss = self.criterion(predicted_noise, noise) * traj_conf.mean()  # 修改：这里乘了mean

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))

        if self.lr > self.cfg.lr_limit and self.iter % self.cfg.lr_decay_every == 0:
            self.lr = self.lr * self.cfg.lr_decay_rate
            print("The LR now:", self.lr)
            # 古法·手调学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model, self.diffusion, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))
                             # global_max=self.dataset['test'].getGlobalMax())

        # if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:  # 修改：取消多模态
        #     if self.cfg.ema is True:
        #         compute_stats(self.diffusion, self.multimodal_dict, self.ema_model, self.logger, self.cfg)
        #     else:
        #         compute_stats(self.diffusion, self.multimodal_dict, self.model, self.logger, self.cfg)

        if self.cfg.save_model_interval > 0 and self.iter % self.cfg.save_model_interval == 0:
            # 修改：检查点中断重读机制
            if self.cfg.ema is True:
                torch.save({
                    'net_state_dict': self.ema_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr': self.lr,
                    'iter': self.iter},
                     os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter}.pt"))
            else:
                torch.save({
                    'net_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr': self.lr,
                    'iter': self.iter},
                    os.path.join(self.cfg.model_path, f"ckpt_{self.iter}.pt"))
