#%%
from dataset import get_dataset_and_dataloader
import torch
import torch.nn.functional as F
import torch_tensorrt
import math
import numpy as np
from pathlib import Path
import yaml
from einops import repeat, rearrange
from model.transformer import Transformer, ConditionNetwork
from utils.plot_signal import plot_signal
from timeit import default_timer as timer
from torchsummary import summary
import wandb
from model.estimator import Estimator
from torch.optim.lr_scheduler import CyclicLR
from transformers import get_cosine_schedule_with_warmup


class Engine:
    def __init__(self, conf_file, device='cuda:0', use_wandb=True, wandb_proj='DNN_channel_estimation'):
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        with open(conf_path, encoding='utf-8') as f:
            self._conf = yaml.safe_load(f)
        self._device = device
        self._use_wandb = use_wandb
        self._wandb_proj = wandb_proj
        if self._use_wandb:
            wandb.init(project=self._wandb_proj, config=self._conf)
            self._conf = wandb.config
        # Get dataset and dataloader
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])
        # Channel and phase noise estimation network
        self._estimator = Estimator(conf_file).to(self._device)
        # Optimizer
        # self._pn_train_start_iter = self._conf['training']['pn_train_start_iter']
        self._max_norm = self._conf['training']['max_norm']
        self._num_iter = self._conf['training']['num_iter']
        self._lr = self._conf['training']['lr']
        self._weight_decay = self._conf['training']['weight_decay']
        self._ch_optimizer = None
        # self._pn_optimizer = None
        self.set_optimizer()

    def set_optimizer(self):
        ch_params = [p for n, p in self._estimator.ch_tf.named_parameters() if p.requires_grad]
        # pn_params = [p for n, p in self._estimator.pn_tf.named_parameters() if p.requires_grad]
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._lr)
        # self._pn_optimizer = torch.optim.Adam([{"params": pn_params}], lr=self._lr)

        # 학습률 스케줄러 사용 여부를 설정 (None이면 사용 안함)
        if self._conf['training'].get('use_scheduler', False):  # 'use_scheduler'가 True일 때만 사용'
            num_warmup_steps = self._conf['training'].get('num_warmup_steps', 0)
            self._ch_scheduler = get_cosine_schedule_with_warmup(
                self._ch_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self._num_iter
            )
        else:
            self._ch_scheduler = None  # 스케줄러 사용 안함

    def train(self):

        # Loss weight 설정
        ch_loss_weight = 1  # 채널 추정 손실 가중치
        # pn_loss_weight = 0.5  # 위상 잡음 추정 손실 가중치

        for it, data in enumerate(self._dataloader):
            # Forward estimator
            self._estimator.train()
            rx_signal = data['ref_comp_rx_signal']
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device)
            # ch_est, pn_est, _ = self._estimator(rx_signal )
            ch_est, _ = self._estimator(rx_signal)
            # Channel training
            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1)  # batch, data, re/im
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1]
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1]
            ch_nmse = torch.mean(ch_mse / ch_var)
            ch_mse = torch.mean(ch_mse)
            # ch_loss = ch_nmse
            ch_loss = ch_nmse * ch_loss_weight  # 채널 추정 손실에 가중치 적용
            self._ch_optimizer.zero_grad()
            ch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._estimator.ch_tf.parameters(), max_norm=self._max_norm)
            self._ch_optimizer.step()

            # 학습률 업데이트 (스케줄러가 있을 때만 실행)
            if self._ch_scheduler:
                self._ch_scheduler.step()

            # # Phase noise training
            # pn_true = torch.tensor(data['pn_time'], dtype=torch.cfloat).to(self._device)
            # pn_true = pn_true * torch.conj(pn_true[:, :1])
            # pn_true = torch.angle(pn_true)  # batch, data
            #
            # pn_mse = torch.sum(torch.square(pn_true - pn_est), dim=1) / pn_true.shape[-1]
            # pn_var = torch.sum(torch.square(pn_true), dim=1) / pn_true.shape[-1]
            # pn_nmse = torch.mean(pn_mse / pn_var)
            # pn_mse = torch.mean(pn_mse)
            # pn_loss = pn_nmse * pn_loss_weight  # 위상 잡음 추정 손실에 가중치 적용
            # if it >= self._pn_train_start_iter:
            #     self._pn_optimizer.zero_grad()
            #     pn_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self._estimator.pn_tf.parameters(), max_norm=self._max_norm)
            #     self._pn_optimizer.step()

            # Logging
            if (it + 1) % self._conf['training']['logging_step'] == 0:
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._lr
                print(f"iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}")
                # self._logging(it, ch_nmse, pn_mse, ch_est, ch_true, pn_est, pn_true)
                self._logging(it, ch_nmse, ch_est, ch_true)

            if it >= self._num_iter - 1:
                break

    @torch.no_grad()
    # def _logging(self, it, ch_nmse, pn_nmse, ch_est, ch_true, pn_est, pn_true):
    def _logging(self, it, ch_nmse, ch_est, ch_true):
        # log = {'ch_nmse': ch_nmse, 'pn_nmse': pn_nmse}
        # print(f"iteration:{it + 1}, ch_nmse:{log['ch_nmse']}, pn_nmse:{log['pn_nmse']}")
        log = {'ch_nmse': ch_nmse}
        # print(f"iteration:{it + 1}, ch_nmse:{log['ch_nmse']}")
        if self._use_wandb:
            wandb.log(log)
        if (it + 1) % self._conf['training']['evaluation_step'] == 0:
            show_batch_size = self._conf['training']['evaluation_batch_size']
            ch_true = ch_true[:, :, 0] + 1j * ch_true[:, :, 1]
            ch_true = ch_true[:show_batch_size].detach().cpu().numpy()
            ch_est = ch_est[:, :, 0] + 1j * ch_est[:, :, 1]
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy()
            # pn_true = pn_true[:show_batch_size].detach().cpu().numpy()
            # pn_est = pn_est[:show_batch_size].detach().cpu().numpy()
            sig_dict = {}
            sig_dict['ch_est_real'] = {'data': ch_est, 'type': 'real'}
            sig_dict['ch_true_real'] = {'data': ch_true, 'type': 'real'}
            sig_dict['ch_est_imag'] = {'data': ch_est, 'type': 'imag'}
            sig_dict['ch_true_imag'] = {'data': ch_true, 'type': 'imag'}
            # sig_dict['pn_est'] = jb yhn{'data': pn_est, 'type': 'scalar'}
            # sig_dict['pn_true'] = {'data': pn_true, 'type': 'scalar'}
            f = plot_signal(sig_dict, shape=(3, 2))
            f.show()
            if self._use_wandb:
                wandb.log({'estimation': wandb.Image(f)})
            self.save_model('Large_estimator_PreLN')
            # self.save_model('estimator')

    def save_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self._estimator, path / (file_name + '.pt'))

    def load_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._estimator = torch.load(path / (file_name + '.pt'))
        self.set_optimizer()

    def num_params(self):
        n_params = sum([p.numel() for p in self._estimator.parameters() if p.requires_grad])
        print(f'total_params: {n_params}')


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    # Training
    conf_file = 'config.yaml'
    engine = Engine(conf_file, device='cuda:0', use_wandb=True, wandb_proj='DNN_channel_estimation')
    # engine.load_model('InF_Nlos_RMa_Large_estimator')
    engine.train()

    # # Model summary
    # conf_file = 'config.yaml'
    # engine = Engine(conf_file, device='cuda:0', use_wandb=False, wandb_proj='DNN_channel_estimation')
    # engine.num_params()

# %%
