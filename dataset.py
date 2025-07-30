import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import scipy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ChannelDataset(IterableDataset):
    def __init__(self, params):
        # load channel pdp data
        self._ch_pdp = []
        for ch_type in params['channel_type']:
            file_path = Path(__file__).parents[0].resolve() / 'dataset' / 'PDP_processed' / ('PDP_' + ch_type + '.mat')
            self._ch_pdp.append(scipy.io.loadmat(file_path)['pdp'][0, :])
        self._ch_pdp = np.concatenate(self._ch_pdp)
        # load phase noise data
        # self._pn = []
        # for pn_type in params['phase_noise_type']:
        #     file_path = Path(__file__).parents[0].resolve() / 'dataset' / 'phase_noise' / ('phase_noise_' + pn_type + '.mat')
        #     self._pn.append(scipy.io.loadmat(file_path)['pn_time'].transpose())
        # self._pn = np.concatenate(self._pn, axis=0)
        # Set parameters
        self._fft_size = params['fft_size']
        self._subcarrier_spacing = params['subcarrier_spacing'] * 1000.0  # Hz
        self._sample_rate = self._fft_size * self._subcarrier_spacing  # Hz
        self._cp_length = params['cp_length'] / 1000000000.0  # second
        self._max_random_tap_delay = params['max_random_tap_delay_cp_proportion'] * self._cp_length  # second
        self._n_subc = self._fft_size - params['num_guard_subcarriers']
        self._subc_start_idx = int(np.ceil(params['num_guard_subcarriers'] / 2))
        self._n_symb = params['num_symbol']
        self._batch_size = params['batch_size']
        noise_spectral_density = np.power(10.0, params['noise_spectral_density'] / 10.0)   # mW/Hz
        self._noise_power = noise_spectral_density * self._subcarrier_spacing  # mW
        self._transmit_power = np.power(10.0, params['transmit_power'] / 10.0)  # mW
        self._distance_range = params['distance_range']  # meter
        c = 299792458.0  # speed of light (m/s)
        carrier_freq = params['carrier_freq'] * 1000000000.0  # Hz
        self._wavelength = c / carrier_freq  # meter
        self._mod_order = params['mod_order']
        self._is_phase_noise = params['is_phase_noise']
        self._is_channel = params['is_channel']
        self._is_noise = params['is_noise']
        self._base_rnd_seed = params['rnd_seed']
        self._rng = None  # random generator
        self._ref_mask_dmrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask_ptrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_signal = None
        self._set_ref_signal(params['ref_conf_dict'])

    @property
    def n_dmrs(self):
        return int(np.sum(self._ref_mask_dmrs))

    @property
    def n_ptrs(self):
        return int(np.sum(self._ref_mask_ptrs))

    def _set_ref_signal(self, ref_conf_dict):
        self._ref_mask_dmrs[0, np.arange(*ref_conf_dict['dmrs'])] = True
        # for s in range(self._n_symb):
        #     self._ref_mask_ptrs[s, np.arange(*ref_conf_dict['ptrs'])] = True
        self._ref_mask = np.logical_or(self._ref_mask_dmrs, self._ref_mask_ptrs)
        ref_num = int(np.sum(self._ref_mask))
        ref_mod_order = int(np.sqrt(4))  # qpsk
        ref_mod_power_correction = np.sqrt(6 / ((ref_mod_order - 1) * (ref_mod_order + 1)))
        tmp_rng = np.random.default_rng(0)
        ref_mod_index = tmp_rng.integers(low=0, high=ref_mod_order, size=(ref_num, 2))
        self._ref_signal = (ref_mod_index - (ref_mod_order - 1) / 2) * ref_mod_power_correction

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = self._base_rnd_seed
        else:
            worker_id = worker_info.id
            seed = self._base_rnd_seed + worker_id
        self._rng = np.random.default_rng(seed)
        return self

    def __next__(self):
        dist = self._rng.uniform(low=self._distance_range[0], high=self._distance_range[1], size=(self._batch_size, 1))
        ch_power_sqrt = np.sqrt(self._transmit_power) * self._wavelength / (4 * np.pi * dist)
        ch_pdp_size = self._ch_pdp.size
        ch_pdp_idx = self._rng.integers(low=0, high=ch_pdp_size, size=(self._batch_size,))
        ch_pdp = self._ch_pdp[ch_pdp_idx]
        ch_time = self._pdp_to_channel(ch_pdp)
        ch_time = ch_time * ch_power_sqrt
        ch_freq = np.fft.fftshift(np.fft.fft(ch_time, axis=-1, norm='ortho'), axes=(-1,))
        ch_freq = ch_freq[:, self._subc_start_idx: self._subc_start_idx + self._n_subc]
        # pn_data_size = self._pn.shape[0]
        # pn_time_idx = self._rng.integers(low=0, high=pn_data_size, size=(self._batch_size,))
        # pn_time = self._pn[pn_time_idx, :]
        noise = self._generate_noise()
        tx_signal = self._generate_tx_signal()
        rx_signal = ch_freq[:, np.newaxis, :] * tx_signal if self._is_channel else tx_signal
        # rx_signal = rx_signal * pn_time[:, :, np.newaxis] if self._is_phase_noise else rx_signal
        rx_signal = rx_signal + noise if self._is_noise else rx_signal
        ref_comp_rx_signal = rx_signal.copy()
        ref_comp_rx_signal[:, self._ref_mask] /= tx_signal[:, self._ref_mask]
        ref_tx_signal_dmrs = tx_signal[:, self._ref_mask_dmrs]
        # ref_tx_signal_ptrs = tx_signal[:, self._ref_mask_ptrs]
        ref_rx_signal_dmrs = rx_signal[:, self._ref_mask_dmrs]
        # ref_rx_signal_ptrs = rx_signal[:, self._ref_mask_ptrs]
        # data = {'ref_mask_dmrs': self._ref_mask_dmrs, 'ref_mask_ptrs': self._ref_mask_ptrs,
        #         'tx_signal': tx_signal, 'rx_signal': rx_signal, 'ref_comp_rx_signal': ref_comp_rx_signal,
        #         'ch_freq': ch_freq, 'ch_time': ch_time, 'pn_time': pn_time, 'noise': noise,
        #         'ref_tx_signal_dmrs': ref_tx_signal_dmrs, 'ref_tx_signal_ptrs': ref_tx_signal_ptrs,
        #         'ref_rx_signal_dmrs': ref_rx_signal_dmrs, 'ref_rx_signal_ptrs': ref_rx_signal_ptrs}
        data = {'ref_mask_dmrs': self._ref_mask_dmrs, 'ref_mask_ptrs': self._ref_mask_ptrs,
                'tx_signal': tx_signal, 'rx_signal': rx_signal, 'ref_comp_rx_signal': ref_comp_rx_signal,
                'ch_freq': ch_freq, 'ch_time': ch_time, 'noise': noise,
                'ref_tx_signal_dmrs': ref_tx_signal_dmrs,
                'ref_rx_signal_dmrs': ref_rx_signal_dmrs}
        return data

    def _pdp_to_channel(self, pdp):
        channel = []
        for p in pdp:
            delay = p[:, 0] / 1000000000.0  # second
            power = p[:, 1]
            random_tap_delay = np.random.rand() * self._max_random_tap_delay
            delay += random_tap_delay
            lim = (delay <= self._cp_length)
            delay = delay[lim]
            power = power[lim]
            num_tap = delay.size
            delay = delay * self._sample_rate
            sample_time = np.arange(self._fft_size)
            time_diff = sample_time[:, np.newaxis] - delay
            sampled_delay = np.sinc(time_diff)
            coef = np.sqrt(power / 2) * (np.random.randn(num_tap) + 1j * np.random.randn(num_tap))
            ch = np.matmul(sampled_delay, coef[:, np.newaxis])[:, 0]
            ch_pow = np.sum(np.square(np.abs(ch)))
            ch = ch / np.sqrt(ch_pow)
            channel.append(ch)
        channel = np.stack(channel, axis=0)
        return channel

    def _generate_noise(self):
        noise = self._rng.normal(loc=0, scale=np.sqrt(self._noise_power / 2), size=(self._batch_size, self._n_symb, self._n_subc, 2))
        noise = noise[..., 0] + 1j * noise[..., 1]
        return noise

    def _generate_tx_signal(self):
        tx_signal = np.zeros((self._batch_size, self._n_symb, self._n_subc, 2))
        ref_num = int(np.sum(self._ref_mask))
        data_num = self._n_symb * self._n_subc - ref_num
        data_mod_order = int(np.sqrt(self._mod_order))
        data_mod_power_correction = np.sqrt(6/((data_mod_order-1)*(data_mod_order+1)))
        data_mod_index = self._rng.integers(low=0, high=data_mod_order, size=(self._batch_size, data_num, 2))
        data_mod = (data_mod_index - (data_mod_order - 1)/2) * data_mod_power_correction
        tx_signal[:, self._ref_mask, :] = self._ref_signal
        tx_signal[:, np.logical_not(self._ref_mask), :] = data_mod
        tx_signal = tx_signal[..., 0] + 1j * tx_signal[..., 1]
        return tx_signal


def get_dataset_and_dataloader(params, is_validation=False):
    num_workers = params.get('num_workers')
    if is_validation:
        dataset = ValidationDataset(params)
    else:
        dataset = ChannelDataset(params)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    return dataset, dataloader

class ValidationDataset(IterableDataset):
    def __init__(self, params):
        self._conf = params # 설정 파라미터 저장
        # 검증 데이터셋 경로를 설정 파일에서 읽어오도록 수정
        self._data_path = Path(__file__).parents[0].resolve() / self._conf.get('validation_data_path', 'sample_data_RMa_Los_withoutPN_100m')
        self._rx_signal_path = self._data_path / 'sample_100m.npy'
        self._true_data_path = self._data_path / 'sample_100m_true.npz'

        # 설정 파일에서 필요한 파라미터 읽어옴
        self._n_symb = self._conf['num_symbol']
        self._fft_size = self._conf['fft_size']
        self._num_guard_subcarriers = self._conf['num_guard_subcarriers']
        self._n_subc = self._fft_size - self._num_guard_subcarriers
        # self._batch_size = self._conf['batch_size'] # 설정 파일에서 batch_size 로드 (실제 데이터의 batch_size와 다를 수 있음)

        # rx_signal 로드 및 형태 변경
        loaded_data = np.load(self._rx_signal_path)

        # 로드된 데이터의 실제 batch_size를 사용
        self._batch_size = loaded_data.shape[0]

        # 로드된 데이터에서 control_signal 부분을 제외하고 ref_comp_rx_signal 부분만 추출
        # control_signal의 크기는 16 * 2 (실수부/허수부) = 32 또는 16 (실수 또는 복소수)
        # export_data_sample 함수에서 np.int32로 저장하므로 실수부/허수부 분리 없이 16개 요소로 가정
        control_signal_size = 16
        ref_comp_rx_signal_flat = loaded_data[:, control_signal_size:]

        # 펼쳐진 ref_comp_rx_signal 데이터를 (batch_size, num_symbol, num_subcarrier, 2) 형태로 reshape
        expected_flat_size_per_batch = self._n_symb * self._n_subc * 2
        if ref_comp_rx_signal_flat.shape[1] != expected_flat_size_per_batch:
             raise ValueError(f"Expected flattened ref_comp_rx_signal size per batch to be {expected_flat_size_per_batch}, but got {ref_comp_rx_signal_flat.shape[1]}")


        self._rx_signal = ref_comp_rx_signal_flat.reshape(self._batch_size, self._n_symb, self._n_subc, 2)

        # 복소수 형태로 변환
        self._rx_signal = self._rx_signal[..., 0] + 1j * self._rx_signal[..., 1]


        true_data = np.load(self._true_data_path)
        self._ch_freq = true_data['ch_freq']
        # self._ref_mask = true_data['ref_mask'] # .npz 파일에 ref_mask가 없으므로 삭제

        # tx_signal은 ref_comp_rx_signal이 이미 계산되어 있으므로 필요 없음
        self._tx_signal = None


        # ref_conf_dict를 사용하여 ref_mask 생성
        self._ref_mask_dmrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask_ptrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        ref_conf_dict = self._conf.get('ref_conf_dict', {}) # 설정에서 ref_conf_dict 가져오기 (기본값 빈 딕셔너리)
        if 'dmrs' in ref_conf_dict:
             self._ref_mask_dmrs[0, np.arange(*ref_conf_dict['dmrs'])] = True
        if 'ptrs' in ref_conf_dict:
             for s in range(self._n_symb):
                  self._ref_mask_ptrs[s, np.arange(*ref_conf_dict['ptrs'])] = True
        self._ref_mask = np.logical_or(self._ref_mask_dmrs, self._ref_mask_ptrs)


    def __iter__(self):
        # For validation, we just yield the entire dataset once
        yield {
            'ref_comp_rx_signal': self._rx_signal, # ref_comp_rx_signal이 이미 계산되어 있으므로 그대로 사용
            'ch_freq': self._ch_freq
        }

    def __next__(self):
        # This should only be called once for the entire dataset
        raise StopIteration


if __name__ == "__main__":
    param_dict = {
        'channel_type': ["RMa_Los_50000"],
        # 'channel_type': ["InF_Los", "InF_Nlos", "InH_Los", "InH_Nlos", "RMa_Los", "RMa_Nlos", "UMa_Los", "UMa_Nlos",
        #                  "UMi_Los", "UMi_Nlos"],
        'phase_noise_type': ["A", "B", "C"],
        'batch_size': 32, # 훈련 시 사용할 batch_size
        'noise_spectral_density': -174.0,  # dBm/Hz
        'subcarrier_spacing': 120.0,  # kHz
        'transmit_power': 30.0,  # dBm
        'distance_range': [5.0, 100.0],  # meter
        'carrier_freq': 28.0,  # GHz
        'mod_order': 64,
        'ref_conf_dict': {'dmrs': (0, 3072, 6)}, # export_data_sample_with_true.py에서 사용된 ref_conf_dict로 변경
        'fft_size': 4096,
        'num_guard_subcarriers': 1024,
        'num_symbol': 14,
        'cp_length': 590,  # cyclic prefix length (ns)
        'max_random_tap_delay_cp_proportion': 0.2,  # random tap delay in proportion of CP length
        'rnd_seed': 0,
        'num_workers': 0,
        'is_phase_noise': False, # export_data_sample_with_true.py에서 사용된 설정으로 변경
        'is_channel': True,
        'is_noise': True,
        'validation_data_path': 'sample_data_RMa_Los_withoutPN_100m' # 검증 데이터셋 경로 추가
    }
    dataset, dataloader = get_dataset_and_dataloader(params=param_dict)
    for it, data in enumerate(dataloader):
        print(it)

    # Example of loading validation dataset
    val_dataset, val_dataloader = get_dataset_and_dataloader(params=param_dict, is_validation=True)
    for it, data in enumerate(val_dataloader):
        print("Validation data loaded")
        print(data['ref_comp_rx_signal'].shape)
        print(data['ch_freq'].shape)
