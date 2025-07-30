from dataset import get_dataset_and_dataloader
import numpy as np
from pathlib import Path


def export_data_sample(data, gain, file_name):
    rx_signal = data['ref_comp_rx_signal']
    batch_size, num_symbol, _ = rx_signal.shape
    # Convert to signed Q10.11
    rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)  # batch, symbol, freq, re/im
    rx_signal = rx_signal * np.power(10.0, gain / 10.0)
    rx_signal = rx_signal * (2.0 ** 11)
    rx_signal = np.round(rx_signal).astype(np.int32)

    # Generate control signal
    control_signal = np.zeros((batch_size, 16)).astype(np.int32)
    control_signal[:, :4] = np.frombuffer(b'\xff\xff\xff\xff', dtype=np.int32)  # preamble 1
    control_signal[:, 4:8] = np.frombuffer(b'\xbe\xbe\xbe\xbe', dtype=np.int32)  # preamble 2
    mod_order = param_dict['mod_order']
    mcs_levels = {4: 1, 16: 3, 64: 5}  # QPSK: 0 or 1, 16QAM: 2 or 3, 64QAM: 4 or 5
    mcs = mcs_levels[mod_order]
    control_signal[:, 9] = mcs  # MCS level
    idx = np.arange(batch_size)
    symbol_num = 0
    slot_num = idx % 8
    subframe_num = (idx // 8) % 10
    frame_num = idx // 80
    control_signal[:, 10] = frame_num * (2 ** 16) + subframe_num
    control_signal[:, 11] = slot_num * (2 ** 16) + symbol_num

    # Combine control signal and rx_signal
    signal = np.concatenate((control_signal, rx_signal.reshape(batch_size, -1)), axis=-1)
    path = Path(__file__).parents[0].resolve() / 'sample_data_InF_withoutPN_50m' / (file_name + '.npy')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, signal)


def export_true_data(data, file_name):
    # 실제 데이터
    ch_freq = data['ch_freq']  # 주파수 도메인 참값 채널 데이터
    # pn_time = data['pn_time']  # 시간 도메인 참값 페이즈 노이즈 데이터

    # 데이터 저장
    path = Path(__file__).parents[0].resolve() / 'sample_data_InF_withoutPN_50m' / (file_name + '_true.npz')
    # 폴더가 없으면 생성
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ch_freq=ch_freq)
    # np.savez(path, ch_freq=ch_freq, pn_time=pn_time)


if __name__ == "__main__":
    param_dict = {
        # 'channel_type': ["RMa_Los_10000"],
        'channel_type': ["InF_Los_10000", "InF_Nlos_50000"],
        # 'channel_type': ["InF_Los", "InF_Nlos", "InH_Los", "InH_Nlos", "RMa_Los", "RMa_Nlos", "UMa_Los", "UMa_Nlos",
        #                  "UMi_Los", "UMi_Nlos"],
        # 'phase_noise_type': ["A", "B", "C"],
        'batch_size': 1024,
        'noise_spectral_density': -174.0,  # dBm/Hz
        'subcarrier_spacing': 120.0,  # kHz
        'transmit_power': 30.0,  # dBm
        # 'distance_range': [10.0, 500.0],  # meter
        'distance_range': [50.0],  # meter
        'carrier_freq': 28.0,  # GHz
        'mod_order': 64,
        # 'ref_conf_dict': {'dmrs': (0, 3072, 1), 'ptrs': (6, 3072, 48)},
        'ref_conf_dict': {'dmrs': (0, 3072, 6)},
        'fft_size': 4096,
        'num_guard_subcarriers': 1024,
        'num_symbol': 14,
        'cp_length': 590,  # cyclic prefix length (ns)
        'max_random_tap_delay_cp_proportion': 0.2,  # random tap delay in proportion of CP length
        'rnd_seed': 0,
        'num_workers': 0,
        'is_phase_noise': False,
        'is_channel': True,
        'is_noise': True
    }

    gain = 66  # dB

    # 거리별 데이터셋 생성
    # for distance in range(10, 500, 80):
    #     print(f"Processing dataset for distance: {distance}m")
    #
    #     # 거리 고정 설정
    #     param_dict['distance_range'] = [distance, distance]
    #
    #     # 데이터셋과 데이터로더 초기화
    #     dataset, dataloader = get_dataset_and_dataloader(params=param_dict)
    #     dl_iter = iter(dataloader)
    #
    #     # 동일한 배치 데이터 가져오기
    #     data = next(dl_iter)
    #
    #     # 기존 데이터 저장
    #     export_data_sample(data, gain, f"sample_{distance}m")
    #
    #     # 실제 데이터 저장
    #     export_true_data(data, f"sample_{distance}m")
    for distance in range(50, 51,1):
        print(f"Processing dataset for distance: {distance}m")

        # 거리 고정 설정
        param_dict['distance_range'] = [distance, distance]

        # 데이터셋과 데이터로더 초기화
        dataset, dataloader = get_dataset_and_dataloader(params=param_dict)
        dl_iter = iter(dataloader)

        # 동일한 배치 데이터 가져오기
        data = next(dl_iter)

        # 기존 데이터 저장
        export_data_sample(data, gain, f"sample_{distance}m")

        # 실제 데이터 저장
        export_true_data(data, f"sample_{distance}m")

    print("Dataset generation complete.")
