from dataset import get_dataset_and_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
import math
import numpy as np
from pathlib import Path
import yaml
from einops import repeat, rearrange
from model.transformer_v3 import Transformer, ConditionNetwork # v3 Transformer 모델 임포트

class Estimator_v3(nn.Module): # Estimator_v3 클래스 정의
    def __init__(self, conf_file):
        super(Estimator_v3, self).__init__()
        conf_path = Path(__file__).parents[1].resolve() / 'config' / conf_file
        with open(conf_path, encoding='utf-8') as f:
            self._conf = yaml.safe_load(f)

        # Channel estimation network (Load pretrained Transformer with Adapters)
        ch_cond_netw = ConditionNetwork(**self._conf['ch_estimation']['cond'])
        # v3 Transformer 모델 인스턴스 생성 시 Adapter 관련 파라미터 전달
        self.ch_tf = Transformer(
            **self._conf['ch_estimation']['transformer'],
            cond_net=ch_cond_netw,
            adapter_bottleneck_dim=self._conf['ch_estimation']['adapter']['bottleneck_dim'],
            adapter_dropout=self._conf['ch_estimation']['adapter']['dropout']
        )

        # Load pretrained weights (excluding Adapter weights)
        pretrained_model_path = Path(__file__).parents[1].resolve() / 'saved_model' / (self._conf['training']['pretrained_model_name'] + '.pt')
        if not pretrained_model_path.exists():
             raise FileNotFoundError(f"Pretrained model file not found at {pretrained_model_path}")

        # Load the pretrained model state dict
        full_state_dict = torch.load(pretrained_model_path)

        # Extract the state dict for the Transformer part (ch_tf)

        # Extract the state dict for the Transformer part (ch_tf)
        # Assuming the Transformer state dict is nested under 'ch_tf' key in the full state dict
        transformer_state_dict_to_load = {}
        for k, v in full_state_dict.items():
            # Check if the key starts with 'ch_tf.'
            if k.startswith('ch_tf.'):
                # Remove the 'ch_tf.' prefix for loading into self.ch_tf
                transformer_state_dict_to_load[k.replace('ch_tf.', '')] = v
            # Handle cases where Transformer parameters might be at the top level or have different prefixes
            # This part might need adjustment based on the actual pretrained model's state dict keys
            # For now, assuming 'ch_tf.' prefix is present or parameters are at the top level.
            elif 'ch_tf' not in k:
                 # Attempt to load parameters that might be directly under the module (e.g., _embedding, _linear)
                 # This is a heuristic and might need refinement.
                 if k in self.ch_tf.state_dict():
                      transformer_state_dict_to_load[k] = v


        if not transformer_state_dict_to_load:
             print("Warning: No Transformer parameters found in the pretrained model state dict with 'ch_tf.' prefix or matching keys.")
             print("Please check the pretrained model file and its state dict keys.")
             # Optionally, raise an error or handle this case based on expected behavior
             # raise ValueError("Could not find Transformer parameters in the pretrained model state dict.")


        # Load the extracted Transformer state dict
        # Use strict=False to ignore keys in the pretrained state dict that are not in the current model
        # (e.g., if the pretrained model had more layers or different structure)
        # Or if the current model has additional layers (like MLP) not in the pretrained model
        try:
            self.ch_tf.load_state_dict(transformer_state_dict_to_load, strict=True)
            print(f"Pretrained Transformer model state dict loaded successfully from {pretrained_model_path}")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Attempting non-strict loading.")
            self.ch_tf.load_state_dict(transformer_state_dict_to_load, strict=False)
            print(f"Pretrained Transformer model state dict loaded non-strictly from {pretrained_model_path}")


        # Freeze all Transformer parameters
        for param in self.ch_tf.parameters():
            param.requires_grad = False

        # Ensure Adapter parameters are trainable
        for name, param in self.ch_tf.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
                print(f"Unfroze adapter parameter: {name}")


        # Phase noise estimation network (주석 처리 유지)
        # pn_cond_netw = ConditionNetwork(**self._conf['pn_estimation']['cond'])
        # self.pn_tf = Transformer(**self._conf['pn_estimation']['transformer'], cond_net=pn_cond_netw)
        # DMRS and PTRS
        self._n_subc = self._conf['dataset']['fft_size'] - self._conf['dataset']['num_guard_subcarriers']
        self._n_symb = self._conf['dataset']['num_symbol']
        self._dmrs_idx = np.arange(*self._conf['dataset']['ref_conf_dict']['dmrs'])
        # self._ptrs_idx = np.arange(*self._conf['dataset']['ref_conf_dict']['ptrs'])

    def forward(self, rx_signal):  # rx_signal: (batch, 14, 3072, 2)
        rx_signal = rx_signal.movedim(source=(0, 1, 2, 3), destination=(0, 2, 3, 1))  #  (batch, 2, 14, 3072)
        # Channel estimation
        # 빈공간에 0을 채워넣는 로직 추가
        ch_in = torch.zeros((rx_signal.size(0), 2, 3072), device=rx_signal.device)  # Create zeros for all subcarriers
        ch_in[:, :, self._dmrs_idx] = rx_signal[:, :, 0, self._dmrs_idx]  # Place DMRS values
        # ch_in = rx_signal[:, :, 0, self._dmrs_idx]  # (batch, re/im, n_dmrs) (batch, 2, 3072)
        ch_std = torch.sqrt(torch.sum(torch.square(ch_in), dim=(1, 2), keepdim=True) / self._n_subc)  # batch, 1, 1
        ch_in = ch_in / ch_std

        # Pass through the Transformer with Adapters
        ch_est = self.ch_tf(ch_in) # (batch, channels, n_token * step_size)
        ch_est = ch_est * ch_std   # (batch, re/im, subc)


        # Channel compensation (No complex number. ((xa + yb) + (-xb + ya)j)/(a^2 + b^2)
        rx_sig_re = rx_signal[:, 0, :, :]  # batch, symbol, subc
        rx_sig_im = rx_signal[:, 1, :, :]  # batch, symbol, subc
        ch_re = ch_est.detach()[:, 0, None, :]  # batch, symbol, subc
        ch_im = ch_est.detach()[:, 1, None, :]  # batch, symbol, subc
        denom = torch.square(ch_re) + torch.square(ch_im)
        rx_sig_comp_re = (rx_sig_re * ch_re + rx_sig_im * ch_im) / denom
        rx_sig_comp_im = (-rx_sig_re * ch_im + rx_sig_im * ch_re) / denom
        rx_sig_comp = torch.stack((rx_sig_comp_re, rx_sig_comp_im), dim=1).detach()  # batch, re/im, symbol, subc

        # # Phase noise estimation
        # pn_in = rx_sig_comp[:, :, :, self._ptrs_idx].flatten(2, 3)  # (batch, re/im, n_ptrs) (batch, 2, 896)
        # pn_est = self.pn_tf(pn_in)[:, 0, :]  # batch, symbol
        #
        # # Phase noise compensation (No complex number. ((xa + yb) + (-xb + ya)j)
        # rx_sig_re = rx_sig_comp[:, 0, :, :]  # batch, symbol, subc
        # rx_sig_im = rx_sig_comp[:, 1, :, :]  # batch, subc
        # pn_re = torch.cos(pn_est.detach())[:, :, None]  # batch, symbol, subc
        # pn_im = torch.sin(pn_est.detach())[:, :, None]  # batch, symbol, subc
        # rx_sig_comp_re = rx_sig_re * pn_re + rx_sig_im * pn_im
        # rx_sig_comp_im = -rx_sig_re * pn_im + rx_sig_im * pn_re
        # rx_sig_comp = torch.stack((rx_sig_comp_re, rx_sig_comp_im), dim=-1).detach()  # batch, symbol, subc, re/im
        ch_est = ch_est.transpose(dim0=1, dim1=2)  # (batch, subc, re/im)

        return ch_est, rx_sig_comp    # , pn_est


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    param_dict = {
        'channel_type': ["InF_Los", "InF_Nlos", "InH_Los", "InH_Nlos", "RMa_Los", "RMa_Nlos", "UMa_Los", "UMa_Nlos",
                         "UMi_Los", "UMi_Nlos"],
        'phase_noise_type': ["A", "B", "C"],
        'batch_size': 32,
        'noise_spectral_density': -174.0,  # dBm/Hz
        'subcarrier_spacing': 120.0,  # kHz
        'transmit_power': 30.0,  # dBm
        'distance_range': [5.0, 30.0],  # meter
        'carrier_freq': 28.0,  # GHz
        'mod_order': 64,
        'ref_conf_dict': {'dmrs': (0, 3072, 1), 'ptrs': (6, 3072, 48)},
        'fft_size': 4096,
        'num_guard_subcarriers': 1024,
        'num_symbol': 14,
        'cp_length': 590,  # cyclic prefix length (ns)
        'max_random_tap_delay_cp_proportion': 0.1,  # random tap delay in proportion of CP length
        'rnd_seed': 0,
        'num_workers': 0,
        'is_phase_noise': False,
        'is_channel': True,
        'is_noise': True
    }
    dataset, dataloader = get_dataset_and_dataloader(params=param_dict)
    conf_file = 'config.yaml'
    estimator = Estimator_v3(conf_file).cuda() # Estimator_v3로 변경
    for it, data in enumerate(dataloader):
        rx_signal = data['ref_comp_rx_signal']
        rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
        rx_signal = torch.tensor(rx_signal, dtype=torch.float32).cuda()
        ch_est, rx_sig_comp = estimator(rx_signal) # pn_est 제거