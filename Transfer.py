import torch # PyTorch 라이브러리 임포트
import torch.nn.functional as F # PyTorch의 함수형 API 임포트
import yaml # YAML 파일 파싱을 위한 라이브러리 임포트
from pathlib import Path # 파일 경로 관리를 위한 Path 객체 임포트
import wandb # Weights & Biases 로깅 라이브러리 임포트
from dataset import get_dataset_and_dataloader # 데이터셋 및 데이터로더를 가져오는 함수 임포트
from model.estimator import Estimator # 모델 Estimator 클래스 임포트
from torch.optim.lr_scheduler import CyclicLR # CyclicLR 스케줄러 임포트
import numpy as np # NumPy 라이브러리 임포트
from utils.plot_signal import plot_signal # 신호 플롯팅 함수 임포트

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.""" # 검증 손실이 일정 기간 동안 개선되지 않으면 훈련을 조기 중단합니다.
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. # 검증 손실이 마지막으로 개선된 후 얼마나 기다릴지 설정합니다.
                            Default: 7 # 기본값: 7
            verbose (bool): If True, prints a message for each validation loss improvement. # True이면 검증 손실 개선 시 메시지를 출력합니다.
                            Default: False # 기본값: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. # 개선으로 간주하기 위한 최소 변화량입니다.
                            Default: 0 # 기본값: 0
            path (str): Path for the checkpoint to be saved to. # 체크포인트 모델을 저장할 경로입니다.
                        Default: 'checkpoint.pt' # 기본값: 'checkpoint.pt'
            trace_func (function): trace print function. # 로그 출력 함수입니다.
                                    Default: print # 기본값: print
        """
        self.patience = patience # 설정된 patience 값
        self.verbose = verbose # 상세 출력 여부
        self.counter = 0 # 개선되지 않은 에포크/이터레이션 카운터
        self.best_score = None # 현재까지의 최고 점수 (음수 검증 손실)
        self.early_stop = False # early stopping 플래그
        self.val_loss_min = np.Inf # 최소 검증 손실 (초기값 무한대)
        self.delta = delta # 최소 개선 변화량
        self.path = path # 체크포인트 저장 경로
        self.trace_func = trace_func # 로그 출력 함수

    def __call__(self, val_loss, model):
        # 검증 손실을 기반으로 점수 계산 (손실이 낮을수록 점수 높음)
        score = -val_loss

        # 첫 번째 검증 단계
        if self.best_score is None:
            self.best_score = score # 현재 점수를 최고 점수로 설정
            self.save_checkpoint(val_loss, model) # 모델 체크포인트 저장
        # 현재 점수가 최고 점수 + delta보다 작으면 개선되지 않음
        elif score < self.best_score + self.delta:
            self.counter += 1 # 카운터 증가
            self.trace_func(f'EarlyStopping counter: {self.counter} of {self.patience}') # 카운터 상태 출력
            # 카운터가 patience에 도달하면 early stop 설정
            if self.counter >= self.patience:
                self.early_stop = True
        # 현재 점수가 최고 점수 + delta보다 크거나 같으면 개선됨
        else:
            self.best_score = score # 최고 점수 업데이트
            self.save_checkpoint(val_loss, model) # 모델 체크포인트 저장
            self.counter = 0 # 카운터 초기화

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.''' # 검증 손실이 감소할 때 모델을 저장합니다.
        if self.verbose: # 상세 출력 모드이면
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...') # 검증 손실 감소 메시지 출력
        torch.save(model.state_dict(), self.path) # 모델의 상태 사전을 지정된 경로에 저장
        self.val_loss_min = val_loss # 최소 검증 손실 업데이트

class TransferLearningEngine:
    def __init__(self, conf_file):
        # 설정 파일 경로 설정
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        # 설정 파일 로드
        with open(conf_path, encoding='utf-8') as f:
            self._conf = yaml.safe_load(f)

        # 설정 파일에서 기본 파라미터 로드
        self._device = self._conf['training'].get('device', 'cuda:0') # 사용할 디바이스 설정 (기본값 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True) # WandB 사용 여부 (기본값 True)
        self._wandb_proj = self._conf['training'].get('wandb_proj', 'DNN_channel_estimation') # WandB 프로젝트 이름 (기본값 'DNN_channel_estimation')


        # WandB 초기화
        if self._use_wandb:
            wandb.init(project=self._wandb_proj, config=self._conf)
            self._conf = wandb.config # WandB config로 업데이트

        # 훈련 데이터셋 및 데이터로더 가져오기 (설정에 따라 RMa 데이터셋 로드)
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])
        # 검증 데이터셋 및 데이터로더 가져오기
        self._val_dataset, self._val_dataloader = get_dataset_and_dataloader(self._conf['dataset'], is_validation=True)

        # 채널 및 위상 잡음 추정 네트워크 (모델은 나중에 로드)

        # 옵티마이저 및 스케줄러는 모델 로드 후 설정

    def load_model(self):
        # 모델 저장 경로 설정
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        file_name = self._conf['training'].get('pretrained_model_name') # 설정 파일에서 모델 이름 로드
        model_path = path / (file_name + '.pt') # 모델 파일 경로
        # 모델 파일 존재 확인
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}") # 파일이 없으면 에러 발생
        self._estimator = torch.load(model_path).to(self._device) # 모델 로드 및 디바이스에 할당
        print(f"Model loaded successfully from {model_path}") # 모델 로드 성공 메시지 출력

        # Freeze initial layers
        num_freeze_layers = self._conf['training'].get('num_freeze_layers') # 설정 파일에서 프리징할 레이어 수 로드
        if num_freeze_layers > 0:
            for i, layer in enumerate(self._estimator.ch_tf._layers):
                if i < num_freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    print(f"Froze layer {i}")
                else:
                    break

        self.set_optimizer() # 옵티마이저 설정

        # Early Stopping 설정
        self._early_stopping = None # early stopping 객체 초기화
        # 설정에서 early stopping 사용 여부 확인
        if self._conf['training'].get('use_early_stopping', False):
            self._early_stopping = EarlyStopping( # EarlyStopping 객체 생성
                patience=self._conf['training'].get('patience', 7), # patience 설정 (기본값 7)
                verbose=self._conf['training'].get('early_stopping_verbose', False), # 상세 출력 활성화 (설정 파일에서 로드)
                delta=self._conf['training'].get('delta', 0), # delta 설정 (기본값 0)
                path=self._conf['training'].get('checkpoint_path', 'checkpoint.pt') # 체크포인트 경로 설정 (기본값 'checkpoint.pt')
            )

    def set_optimizer(self):
        # 전이 학습을 위해 채널 추정기 부분만 훈련한다고 가정
        ch_params = [p for n, p in self._estimator.ch_tf.named_parameters() if p.requires_grad] # 훈련 가능한 채널 추정기 파라미터 가져오기
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr']) # Adam 옵티마이저 설정

        # 스케줄러 사용 여부 확인
        if self._conf['training'].get('use_scheduler', False):
            self._ch_scheduler = CyclicLR( # CyclicLR 스케줄러 설정
                self._ch_optimizer, # 옵티마이저
                base_lr=self._conf['training']['base_lr'], # 최소 학습률
                max_lr=self._conf['training']['max_lr'], # 최대 학습률
                step_size_up=self._conf['training']['step_size_up'], # 학습률 증가 주기
                mode='triangular2', # 스케줄러 모드
            )
        else:
            self._ch_scheduler = None # 스케줄러 사용 안 함

    def train(self):
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1) # 채널 손실 가중치 (설정 파일에서 로드, 기본값 1)

        # 훈련 데이터로더를 순회하며 훈련
        for it, data in enumerate(self._dataloader):
            self._estimator.train() # 모델을 훈련 모드로 설정
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산
            ch_mse = torch.mean(ch_mse) # 채널 MSE 평균 계산
            ch_loss = ch_nmse * ch_loss_weight # 채널 손실 계산 (NMSE에 가중치 적용)

            self._ch_optimizer.zero_grad() # 옵티마이저의 그래디언트 초기화
            ch_loss.backward() # 역전파를 통해 그래디언트 계산
            torch.nn.utils.clip_grad_norm_(self._estimator.ch_tf.parameters(), max_norm=self._conf['training']['max_norm']) # 그래디언트 클리핑
            self._ch_optimizer.step() # 옵티마이저 스텝 (파라미터 업데이트)

            # 스케줄러 사용 시 학습률 업데이트
            if self._ch_scheduler:
                self._ch_scheduler.step()

            # 로깅 스텝마다 정보 출력 및 로깅
            if (it + 1) % self._conf['training']['logging_step'] == 0:
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._conf['training']['lr'] # 현재 학습률 가져오기
                print(f"iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}") # 훈련 상태 출력
                self._logging(it, ch_nmse, ch_est, ch_true) # 로깅 함수 호출

            # 검증 및 Early Stopping 체크
            if (it + 1) % self._conf['training']['evaluation_step'] == 0: # 평가 스텝마다
                if self._early_stopping: # early stopping 사용 시
                    val_loss = self.evaluate() # 검증 손실 계산
                    self._early_stopping(val_loss, self._estimator) # early stopping 객체에 검증 손실 전달
                    if self._early_stopping.early_stop: # early stopping 조건 충족 시
                        print("Early stopping") # 메시지 출력
                        break # 훈련 루프 중단


            # 설정된 최대 이터레이션에 도달하면 훈련 중단
            if it >= self._conf['training']['num_iter'] - 1:
                break

        # early stopping으로 훈련이 중단된 경우 최적 모델 로드
        if self._early_stopping and self._early_stopping.early_stop:
             self._estimator.load_state_dict(torch.load(self._early_stopping.path)) # 저장된 최적 모델 상태 로드
             print(f"Loaded best model from {self._early_stopping.path}") # 로드 메시지 출력


    @torch.no_grad() # 그래디언트 계산 비활성화
    def evaluate(self):
        self._estimator.eval() # 모델을 평가 모드로 설정
        total_nmse = 0.0 # 총 NMSE 초기화
        num_batches = 0 # 배치 카운터 초기화
        # 검증 데이터로더를 순회하며 평가
        for data in self._val_dataloader:
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산

            total_nmse += ch_nmse.item() # 총 NMSE에 현재 배치의 NMSE 추가
            num_batches += 1 # 배치 카운터 증가

        avg_nmse = total_nmse / num_batches # 평균 NMSE 계산
        print(f"Validation NMSE: {avg_nmse}") # 검증 NMSE 출력
        if self._use_wandb: # WandB 사용 시
            wandb.log({'val_ch_nmse': avg_nmse}) # 검증 NMSE 로깅
        return avg_nmse # 평균 NMSE 반환


    @torch.no_grad() # 그래디언트 계산 비활성화
    def _logging(self, it, ch_nmse, ch_est, ch_true):
        log = {'ch_nmse': ch_nmse} # 훈련 NMSE 로깅 데이터
        if self._use_wandb: # WandB 사용 시
            wandb.log(log) # 훈련 NMSE 로깅
        if (it + 1) % self._conf['training']['evaluation_step'] == 0: # 평가 스텝마다
            show_batch_size = self._conf['training']['evaluation_batch_size'] # 플롯팅할 배치 크기
            ch_true = ch_true[:, :, 0] + 1j * ch_true[:, :, 1] # 실제 채널 복소수 형태로 변환
            ch_true = ch_true[:show_batch_size].detach().cpu().numpy() # 플롯팅할 실제 채널 데이터 (CPU로 이동 및 NumPy 변환)
            ch_est = ch_est[:, :, 0] + 1j * ch_est[:, :, 1] # 추정 채널 복소수 형태로 변환
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy() # 플롯팅할 추정 채널 데이터 (CPU로 이동 및 NumPy 변환)

            sig_dict = {} # 신호 플롯팅을 위한 딕셔너리
            sig_dict['ch_est_real'] = {'data': ch_est, 'type': 'real'} # 추정 채널 실수부
            sig_dict['ch_true_real'] = {'data': ch_true, 'type': 'real'} # 실제 채널 실수부
            sig_dict['ch_est_imag'] = {'data': ch_est, 'type': 'imag'} # 추정 채널 허수부
            sig_dict['ch_true_imag'] = {'data': ch_true, 'type': 'imag'} # 실제 채널 허수부 (ch_imag를 ch_true로 수정)

            f = plot_signal(sig_dict, shape=(3, 2)) # 신호 플롯 생성
            f.show() # 플롯 표시
            if self._use_wandb: # WandB 사용 시
                wandb.log({'estimation': wandb.Image(f)}) # 플롯 이미지를 WandB에 로깅
            # 모델 저장 파일 이름을 설정 파일에서 읽어오도록 수정
            self.save_model(self._conf['training'].get('saved_model_name', 'checkpoint'))

    def save_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model' # 모델 저장 디렉토리 경로
        path.mkdir(parents=True, exist_ok=True) # 디렉토리가 없으면 생성
        torch.save(self._estimator, path / (file_name + '.pt')) # 모델 저장
        print(f"Model saved to {path / (file_name + '.pt')}") # 모델 저장 경로 출력


if __name__ == "__main__":
    conf_file = 'config_transfer.yaml' # 설정 파일 이름
    engine = TransferLearningEngine(conf_file) # TransferLearningEngine 객체 생성 (파라미터는 설정 파일에서 로드)
    engine.load_model() # 사전 훈련된 모델 로드 (이름은 설정 파일에서 로드)
    engine.train() # 훈련 시작