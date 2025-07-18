import yaml
import os

from utils import util, torch, generate_pad


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    index = get_log_dir_index(cfg.base_dir)
    if args_dict['mode'] == ('train' or 'pred' or 'eval'):
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['cfg'] + index)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'] + index)
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    return cfg


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        # cfg_name = './Diffusion_Model_wzj/cfg/%s.yml' % cfg_id
        cfg_name = os.path.join(os.getcwd(),'Diffusion_Model_wzj', 'cfg', '%s.yml' % cfg_id)
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        # cfg = yaml.safe_load(open(cfg_name, 'r'))
        # 修改 config.py 中的文件读取方式，让它可以读中文
        with open(cfg_name, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # create dirs
        self.base_dir = 'inference' if test else 'results'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg['batch_size']
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']
        self.lr_decay_every = cfg['lr_decay_every']
        self.lr_decay_rate = cfg['lr_decay_rate']
        self.lr_limit = cfg['lr_limit']

        self.n_pre = cfg['n_pre']
        self.multimodal_path = cfg['multimodal_path']
        self.data_candi_path = cfg['data_candi_path']
        self.train_path = cfg['train_path']
        self.test_path = cfg['test_path']
        self.train_path_imu = cfg['train_path_imu']
        self.test_path_imu = cfg['test_path_imu']

        self.padding = cfg['padding']
        self.Complete = cfg['Complete']
        self.noise_steps = cfg['noise_steps']
        self.ddim_timesteps = cfg['ddim_timesteps']
        self.scheduler = cfg['scheduler']

        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']

        self.dct_norm_enable = cfg['dct_norm_enable']

        # indirect variable
        if self.dataset == 'h36m':
            self.joint_num = 16
        elif self.dataset == 'humaneva':
            self.joint_num = 14
        elif self.dataset == 'xrf2':
            self.joint_num = 15
        else:
            print("你忘了配置关节数量了")

        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
