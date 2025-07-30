from dataset import get_dataset_and_dataloader # 데이터셋 및 데이터로더를 가져오는 함수 임포트
import torch # PyTorch 라이브러리 임포트
import torch.nn as nn # 신경망 모듈 임포트
import torch.nn.functional as F # 함수형 API 임포트
import torch_tensorrt # Torch-TensorRT 임포트 (모델 최적화용, 현재 사용되지 않음)
import math # 수학 함수 임포트
import numpy as np # NumPy 라이브러리 임포트 (수치 연산용)
from pathlib import Path # 파일 경로 관리를 위한 Path 객체 임포트
import yaml # YAML 파일 파싱을 위한 라이브러리 임포트
from einops import repeat, rearrange # einops 라이브러리 임포트 (텐서 재배열용, 현재 주석 처리됨)
from model.transformer_v4 import Transformer, ConditionNetwork # v4 Transformer 모델 임포트

class Estimator_v4(nn.Module): # Estimator_v4 클래스 정의 (채널 추정 모델)
    def __init__(self, conf_file): # 초기화 메서드
        super(Estimator_v4, self).__init__() # 부모 클래스 초기화
        conf_path = Path(__file__).parents[1].resolve() / 'config' / conf_file # 설정 파일 경로 설정
        with open(conf_path, encoding='utf-8') as f: # 설정 파일 열기
            self._conf = yaml.safe_load(f) # 설정 파일 로드

        # Channel estimation network (Load pretrained Transformer with LoRA) # 채널 추정 네트워크 (LoRA가 적용된 사전 학습된 트랜스포머 로드)
        ch_cond_netw = ConditionNetwork(**self._conf['ch_estimation']['cond']) # ConditionNetwork 인스턴스 생성 (설정 파일에서 파라미터 로드)
        # v4 Transformer 모델 인스턴스 생성
        self.ch_tf = Transformer( # Transformer 모델 인스턴스 생성
            **self._conf['ch_estimation']['transformer'], # 설정 파일에서 트랜스포머 파라미터 로드
            cond_net=ch_cond_netw # ConditionNetwork 인스턴스 전달
        )

        # Load pretrained weights # 사전 학습된 가중치 로드
        pretrained_model_path = Path(__file__).parents[1].resolve() / 'saved_model' / (self._conf['training']['pretrained_model_name'] + '.pt') # 사전 학습된 모델 파일 경로 설정
        if not pretrained_model_path.exists(): # 모델 파일이 존재하는지 확인
             raise FileNotFoundError(f"Pretrained model file not found at {pretrained_model_path}") # 파일이 없으면 예외 발생

        # Load the pretrained model state dict # 사전 학습된 모델의 상태 사전 로드
        full_state_dict = torch.load(pretrained_model_path) # 모델 상태 사전 로드

        # Extract the state dict for the Transformer part (ch_tf) # 트랜스포머 부분의 상태 사전 추출
        transformer_state_dict_to_load = {} # 로드할 트랜스포머 상태 사전 딕셔너리 초기화
        # full_state_dict가 모델 객체인 경우 state_dict()를 호출하여 실제 상태 사전을 가져옵니다.
        if hasattr(full_state_dict, 'state_dict'):
            full_state_dict = full_state_dict.state_dict()
        for k, v in full_state_dict.items(): # 전체 상태 사전의 각 키-값 쌍 순회
            # Check if the key starts with 'ch_tf.' # 키가 'ch_tf.'로 시작하는지 확인
            if k.startswith('ch_tf.'):
                # Remove the 'ch_tf.' prefix for loading into self.ch_tf # 'ch_tf.' 접두사 제거
                transformer_state_dict_to_load[k.replace('ch_tf.', '')] = v # 추출된 상태 사전에 추가
            # Handle cases where Transformer parameters might be at the top level or have different prefixes # 트랜스포머 파라미터가 최상위 레벨에 있거나 다른 접두사를 가지는 경우 처리
            elif 'ch_tf' not in k: # 키에 'ch_tf'가 포함되지 않으면
                 if k in self.ch_tf.state_dict(): # 현재 트랜스포머 모델의 상태 사전에 키가 있으면
                      transformer_state_dict_to_load[k] = v # 추출된 상태 사전에 추가


        if not transformer_state_dict_to_load: # 로드할 트랜스포머 파라미터가 없으면
             print("Warning: No Transformer parameters found in the pretrained model state dict with 'ch_tf.' prefix or matching keys.") # 경고 메시지 출력
             print("Please check the pretrained model file and its state dict keys.") # 확인 요청 메시지 출력


        # Load the extracted Transformer state dict # 추출된 트랜스포머 상태 사전 로드
        try:
            self.ch_tf.load_state_dict(transformer_state_dict_to_load, strict=False) # 추출된 상태 사전 로드 (strict=False로 부분 로드 허용)
            print(f"Pretrained Transformer model state dict loaded successfully from {pretrained_model_path}") # 로드 성공 메시지 출력
        except RuntimeError as e: # 로드 중 RuntimeError 발생 시
            print(f"Error loading pretrained model state dict: {e}") # 오류 메시지 출력


        # PEFT 라이브러리를 사용하여 LoRA 어댑터 주입 및 파라미터 프리징은 Transfer_v4.py에서 처리
        # Estimator_v4에서는 기본 Transformer 모델을 로드하고, PEFT 적용은 외부에서 수행


        # Phase noise estimation network (주석 처리 유지) # 위상 잡음 추정 네트워크 (주석 처리됨)
        # pn_cond_netw = ConditionNetwork(**self._conf['pn_estimation']['cond']) # ConditionNetwork 인스턴스 생성 (주석 처리됨)
        # self.pn_tf = Transformer(**self._conf['pn_estimation']['transformer'], cond_net=pn_cond_netw) # Transformer 인스턴스 생성 (주석 처리됨)
        # DMRS and PTRS # DMRS 및 PTRS 관련 설정
        self._n_subc = self._conf['dataset']['fft_size'] - self._conf['dataset']['num_guard_subcarriers'] # 서브캐리어 수 계산
        self._n_symb = self._conf['dataset']['num_symbol'] # 심볼 수 저장
        self._dmrs_idx = np.arange(*self._conf['dataset']['ref_conf_dict']['dmrs']) # DMRS 인덱스 생성
        # self._ptrs_idx = np.arange(*self._conf['dataset']['ref_conf_dict']['ptrs']) # PTRS 인덱스 생성 (주석 처리됨)

    def forward(self, rx_signal):  # rx_signal: (batch, 14, 3072, 2) # 순전파 메서드 (입력: 수신 신호)
        rx_signal = rx_signal.movedim(source=(0, 1, 2, 3), destination=(0, 2, 3, 1))  #  (batch, 2, 14, 3072) # 텐서 차원 재배열
        # Channel estimation # 채널 추정
        # 빈공간에 0을 채워넣는 로직 추가 # 빈 공간에 0을 채워넣는 로직 추가
        ch_in = torch.zeros((rx_signal.size(0), 2, 3072), device=rx_signal.device)  # Create zeros for all subcarriers # 모든 서브캐리어에 대해 0으로 채워진 텐서 생성
        ch_in[:, :, self._dmrs_idx] = rx_signal[:, :, 0, self._dmrs_idx]  # Place DMRS values # DMRS 값 배치
        # ch_in = rx_signal[:, :, 0, self._dmrs_idx]  # (batch, re/im, n_dmrs) (batch, 2, 3072) # DMRS 부분만 가져오기 (주석 처리됨)
        ch_std = torch.sqrt(torch.sum(torch.square(ch_in), dim=(1, 2), keepdim=True) / self._n_subc)  # batch, 1, 1 # 채널 표준 편차 계산
        ch_in = ch_in / ch_std # 채널 입력 정규화

        # Pass through the Transformer with LoRA # LoRA가 적용된 트랜스포머 통과
        ch_est = self.ch_tf(ch_in) # (batch, channels, n_token * step_size) # 트랜스포머를 통해 채널 추정
        ch_est = ch_est * ch_std   # (batch, re/im, subc) # 추정된 채널 역정규화


        # Channel compensation (No complex number. ((xa + yb) + (-xb + ya)j)/(a^2 + b^2) # 채널 보상 (복소수 사용 안 함)
        rx_sig_re = rx_signal[:, 0, :, :]  # batch, symbol, subc # 수신 신호 실수부 가져오기
        rx_sig_im = rx_signal[:, 1, :, :]  # batch, symbol, subc # 수신 신호 허수부 가져오기
        ch_re = ch_est.detach()[:, 0, None, :]  # batch, symbol, subc # 추정 채널 실수부 가져오기 (detach하여 그래디언트 계산 제외)
        ch_im = ch_est.detach()[:, 1, None, :]  # batch, symbol, subc # 추정 채널 허수부 가져오기 (detach하여 그래디언트 계산 제외)
        denom = torch.square(ch_re) + torch.square(ch_im) # 분모 계산 (채널 크기의 제곱)
        rx_sig_comp_re = (rx_sig_re * ch_re + rx_sig_im * ch_im) / denom # 보상된 수신 신호 실수부 계산
        rx_sig_comp_im = (-rx_sig_re * ch_im + rx_sig_im * ch_re) / denom # 보상된 수신 신호 허수부 계산
        rx_sig_comp = torch.stack((rx_sig_comp_re, rx_sig_comp_im), dim=1).detach()  # batch, re/im, symbol, subc # 보상된 수신 신호 텐서 생성 (detach)

        # # Phase noise estimation # 위상 잡음 추정 (주석 처리됨)
        # pn_in = rx_sig_comp[:, :, :, self._ptrs_idx].flatten(2, 3)  # (batch, re/im, n_ptrs) (batch, 2, 896) # PTRS 부분 가져와 평탄화 (주석 처리됨)
        # pn_est = self.pn_tf(pn_in)[:, 0, :]  # batch, symbol # 위상 잡음 추정 (주석 처리됨)
        #
        # # Phase noise compensation (No complex number. ((xa + yb) + (-xb + ya)j) # 위상 잡음 보상 (복소수 사용 안 함) (주석 처리됨)
        # rx_sig_re = rx_sig_comp[:, 0, :, :]  # batch, symbol, subc # 보상된 수신 신호 실수부 (주석 처리됨)
        # rx_sig_im = rx_sig_comp[:, 1, :, :]  # batch, subc # 보상된 수신 신호 허수부 (주석 처리됨)
        # pn_re = torch.cos(pn_est.detach())[:, :, None]  # batch, symbol, subc # 추정 위상 잡음 코사인 값 (detach) (주석 처리됨)
        # pn_im = torch.sin(pn_est.detach())[:, :, None]  # batch, symbol, subc # 추정 위상 잡음 사인 값 (detach) (주석 처리됨)
        # rx_sig_comp_re = rx_sig_re * pn_re + rx_sig_im * pn_im # 위상 잡음 보상된 수신 신호 실수부 (주석 처리됨)
        # rx_sig_comp_im = -rx_sig_re * pn_im + rx_sig_im * pn_re # 위상 잡음 보상된 수신 신호 허수부 (주석 처리됨)
        # rx_sig_comp = torch.stack((rx_sig_comp_re, rx_sig_comp_im), dim=-1).detach()  # batch, symbol, subc, re/im # 위상 잡음 보상된 수신 신호 텐서 생성 (detach) (주석 처리됨)
        ch_est = ch_est.transpose(dim0=1, dim1=2)  # (batch, subc, re/im) # 추정 채널 텐서 차원 변경

        return ch_est, rx_sig_comp    # , pn_est # 추정 채널 및 보상된 수신 신호 반환 (위상 잡음 추정 결과는 주석 처리됨)


if __name__ == "__main__": # 스크립트 직접 실행 시 실행되는 코드 블록
    #torch.autograd.set_detect_anomaly(True) # 자동 미분 이상 감지 활성화 (주석 처리됨)
    param_dict = { # 데이터셋 파라미터 딕셔너리 정의
        'channel_type': ["InF_Los", "InF_Nlos", "InH_Los", "InH_Nlos", "RMa_Los", "RMa_Nlos", "UMa_Los", "UMa_Nlos",
                         "UMi_Los", "UMi_Nlos"], # 채널 타입 목록
        'phase_noise_type': ["A", "B", "C"], # 위상 잡음 타입 목록
        'batch_size': 32, # 배치 크기
        'noise_spectral_density': -174.0,  # dBm/Hz # 잡음 스펙트럼 밀도
        'subcarrier_spacing': 120.0,  # kHz # 서브캐리어 간격
        'transmit_power': 30.0,  # dBm # 전송 파워
        'distance_range': [5.0, 30.0],  # meter # 거리 범위
        'carrier_freq': 28.0,  # GHz # 캐리어 주파수
        'mod_order': 64, # 변조 방식 차수
        'ref_conf_dict': {'dmrs': (0, 3072, 1), 'ptrs': (6, 3072, 48)}, # 참조 신호 설정 딕셔너리
        'fft_size': 4096, # FFT 크기
        'num_guard_subcarriers': 1024, # 가드 서브캐리어 수
        'num_symbol': 14, # 심볼 수
        'cp_length': 590,  # cyclic prefix length (ns) # CP 길이 (ns)
        'max_random_tap_delay_cp_proportion': 0.1,  # random tap delay in proportion of CP length # CP 길이 대비 최대 랜덤 탭 지연 비율
        'rnd_seed': 0, # 랜덤 시드
        'num_workers': 0, # 데이터 로더 워커 수
        'is_phase_noise': False, # 위상 잡음 사용 여부
        'is_channel': True, # 채널 사용 여부
        'is_noise': True # 잡음 사용 여부
    }
    dataset, dataloader = get_dataset_and_dataloader(params=param_dict) # 데이터셋 및 데이터로더 가져오기
    conf_file = 'config_transfer_v4.yaml' # v4 설정 파일 사용
    estimator = Estimator_v4(conf_file).cuda() # Estimator_v4 인스턴스 생성 및 CUDA 디바이스로 이동
    for it, data in enumerate(dataloader): # 데이터로더 순회
        rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
        rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리
        rx_signal = torch.tensor(rx_signal, dtype=torch.float32).cuda() # NumPy 배열을 PyTorch 텐서로 변환 및 CUDA 디바이스로 이동
        ch_est, rx_sig_comp = estimator(rx_signal) # pn_est 제거 # Estimator를 통해 채널 추정 및 수신 신호 보상 (위상 잡음 추정 결과는 제거)