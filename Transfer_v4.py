import torch # PyTorch 라이브러리 임포트 (딥러닝 프레임워크)
import torch.nn.functional as F # PyTorch의 함수형 API 임포트
import yaml # YAML 파일 파싱을 위한 라이브러리 임포트
from pathlib import Path # 파일 경로 관리를 위한 Path 객체 임포트
import wandb # Weights & Biases 로깅 라이브러리 임포트
from dataset import get_dataset_and_dataloader # 데이터셋 및 데이터로더를 가져오는 함수 임포트
from model.estimator_v4 import Estimator_v4 # 모델 Estimator_v4 클래스 임포트
from transformers import get_cosine_schedule_with_warmup # get_cosine_schedule_with_warmup 스케줄러 임포트
import numpy as np # NumPy 라이브러리 임포트 (수치 연산용)
from utils.plot_signal import plot_signal # 신호 플롯팅 함수 임포트
from peft import LoraConfig, get_peft_model # PEFT 라이브러리 임포트
# from model.transformer_v2 import Transformer # v2 Transformer 모델 임포트 (v3에서는 사용 안 함, 주석 처리됨)

class EarlyStopping: # EarlyStopping 클래스 정의 (훈련 조기 중단 기능)
    """Early stops the training if validation loss doesn't improve after a given patience.""" # 검증 손실이 일정 기간 동안 개선되지 않으면 훈련을 조기 중단합니다.
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print): # 초기화 메서드
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
        self.patience = patience # 설정된 patience 값 저장
        self.verbose = verbose # 상세 출력 여부 저장
        self.counter = 0 # 개선되지 않은 에포크/이터레이션 카운터 초기화
        self.best_score = None # 현재까지의 최고 점수 (음수 검증 손실) 초기화
        self.early_stop = False # early stopping 플래그 초기화
        self.val_loss_min = np.Inf # 최소 검증 손실 (초기값 무한대) 초기화
        self.delta = delta # 최소 개선 변화량 저장
        self.path = path # 체크포인트 저장 경로 저장
        self.trace_func = trace_func # 로그 출력 함수 저장

    def __call__(self, val_loss, model): # EarlyStopping 객체를 함수처럼 호출할 때 실행되는 메서드
        # 검증 손실을 기반으로 점수 계산 (손실이 낮을수록 점수 높음)
        score = -val_loss # 검증 손실에 음수를 취하여 점수 계산 (최대화 문제로 변환)

        # 첫 번째 검증 단계
        if self.best_score is None: # 최고 점수가 아직 설정되지 않았으면
            self.best_score = score # 현재 점수를 최고 점수로 설정
            self.save_checkpoint(val_loss, model) # 모델 체크포인트 저장
        # 현재 점수가 최고 점수 + delta보다 작으면 개선되지 않음
        elif score < self.best_score + self.delta: # 현재 점수가 최고 점수 + delta보다 작으면 (개선되지 않았으면)
            self.counter += 1 # 카운터 증가
            self.trace_func(f'EarlyStopping counter: {self.counter} of {self.patience}') # 카운터 상태 출력
            # 카운터가 patience에 도달하면 early stop 설정
            if self.counter >= self.patience: # 카운터가 patience 값 이상이면
                self.early_stop = True # early stop 플래그를 True로 설정
        # 현재 점수가 최고 점수 + delta보다 크거나 같으면 개선됨
        else: # 현재 점수가 최고 점수 + delta보다 크거나 같으면 (개선되었으면)
            self.best_score = score # 최고 점수 업데이트
            self.save_checkpoint(val_loss, model) # 모델 체크포인트 저장
            self.counter = 0 # 카운터 초기화

    def save_checkpoint(self, val_loss, model): # 모델 체크포인트를 저장하는 메서드
        '''Saves model when validation loss decrease.''' # 검증 손실이 감소할 때 모델을 저장합니다.
        if self.verbose: # 상세 출력 모드이면
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...') # 검증 손실 감소 메시지 출력
        # LoRA 가중치를 원본 모델에 병합하여 저장 (engine.py와 호환되는 형태)
        merged_model = model.merge_and_unload()
        torch.save(merged_model, self.path) # 병합된 모델을 지정된 경로에 저장
        self.val_loss_min = val_loss # 최소 검증 손실 업데이트

class TransferLearningEngine: # 전이 학습 엔진 클래스 정의
    def __init__(self, conf_file): # 초기화 메서드
        # 설정 파일 경로 설정
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file # 설정 파일 경로 생성
        # 설정 파일 로드
        with open(conf_path, encoding='utf-8') as f: # 설정 파일 열기
            self._conf = yaml.safe_load(f) # 설정 파일 로드
        self._conf_file = conf_file # 설정 파일 이름 저장

        # 설정 파일에서 기본 파라미터 로드
        self._device = self._conf['training'].get('device', 'cuda:0') # 사용할 디바이스 설정 (기본값 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True) # WandB 사용 여부 (기본값 True)
        self._wandb_proj = self._conf['training'].get('wandb_proj', 'DNN_channel_estimation') # WandB 프로젝트 이름 (기본값 'DNN_channel_estimation')


        # WandB 초기화
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.init(project=self._wandb_proj, config=self._conf) # WandB 초기화 (프로젝트 이름 및 설정 전달)
            self._conf = wandb.config # WandB config로 업데이트 (하이퍼파라미터 스위프 등 고려)
        
        # 훈련 데이터셋 및 데이터로더 가져오기 (설정에 따라 RMa 데이터셋 로드)
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset']) # 훈련 데이터셋 및 데이터로더 가져오기
        # 검증 데이터셋 및 데이터로더 가져오기 (검증 데이터가 없으므로 비활성화)
        self._val_dataset, self._val_dataloader = None, None # 검증 데이터셋 비활성화

        # 채널 및 위상 잡음 추정 네트워크 (모델은 나중에 로드) # 채널 및 위상 잡음 추정 네트워크 (모델은 load_model 메서드에서 로드)

        # 옵티마이저 및 스케줄러는 모델 로드 후 설정 # 옵티마이저 및 스케줄러는 load_model 메서드 호출 후 set_optimizer 메서드에서 설정

    def load_model(self): # 모델 로드 및 초기화 메서드
        # Estimator_v4 클래스는 내부에서 모델 로딩 및 프리징을 처리합니다.
        self._estimator = Estimator_v4(self._conf_file).to(self._device) # Estimator_v4 인스턴스 생성 및 지정된 디바이스로 이동

        # PEFT LoRA 설정 로드 및 모델에 적용
        peft_config_dict = self._conf['ch_estimation'].get('peft', {}) # 설정 파일에서 PEFT 설정 가져오기
        if peft_config_dict: # PEFT 설정이 존재하면
            peft_type = peft_config_dict.get('peft_type', 'LORA') # PEFT 타입 가져오기 (기본값 'LORA')
            if peft_type == 'LORA': # PEFT 타입이 'LORA'이면
                lora_config = LoraConfig( # LoraConfig 객체 생성
                    r=peft_config_dict.get('r', 8), # 랭크 설정 (기본값 8)
                    lora_alpha=peft_config_dict.get('lora_alpha', 16), # alpha 설정 (기본값 16)
                    target_modules=peft_config_dict.get('target_modules', ["q_proj", "k_proj", "v_proj", "out_proj", "ffnn_linear1", "ffnn_linear2"]), # 타겟 모듈 설정
                    lora_dropout=peft_config_dict.get('lora_dropout', 0.1), # 드롭아웃 설정 (기본값 0.1)
                    bias="none", # bias 설정 (LoRA에서는 일반적으로 "none")
                    # task_type을 제거하거나 None으로 설정하여 prepare_inputs_for_generation 요구사항 회피
                    # PEFT가 모델의 기본 동작을 방해하지 않도록 함
                )
                self._estimator = get_peft_model(self._estimator, lora_config) # Estimator_v4 모델에 LoRA 적용
                print("PEFT LoRA model applied to Estimator_v4.") # LoRA 적용 메시지 출력
                self._estimator.print_trainable_parameters() # 훈련 가능한 파라미터 출력

        model_load_mode = self._conf['training'].get('model_load_mode', 'pretrained') # 모델 로드 모드 설정 파일에서 로드 (기본값 'pretrained')

        if model_load_mode == 'finetune': # 모델 로드 모드가 'finetune'이면
            load_path = self._conf['training'].get('load_model_path') # 로드할 모델 경로 설정 파일에서 로드
            if load_path: # 로드 경로가 설정되어 있으면
                try:
                    model_path = Path(load_path) # 로드 경로를 Path 객체로 변환
                    if model_path.exists(): # 모델 파일이 존재하는지 확인
                        # PEFT 모델의 LoRA 가중치만 로드
                        self._estimator.load_state_dict(torch.load(model_path), strict=False) # PEFT 모델에 상태 사전 로드 (strict=False로 부분 로드 허용)
                        print(f"PEFT LoRA weights loaded successfully from {model_path} for finetuning.") # LoRA 가중치 로드 성공 메시지 출력
                    else: # 모델 파일이 존재하지 않으면
                        print(f"Error: Model file not found at {model_path} for finetuning.") # 오류 메시지 출력
                        print("Please check 'load_model_path' in the config file.") # 설정 파일 확인 요청 메시지 출력
                        raise FileNotFoundError(f"Model file not found at {model_path}") # FileNotFoundError 발생
                except Exception as e: # 로드 중 예외 발생 시
                    print(f"Error loading model from {load_path}: {e}") # 오류 메시지 출력
                    print("Please check 'load_model_path' and the model file.") # 경로 및 파일 확인 요청 메시지 출력
                    raise RuntimeError(f"Failed to load model from {load_path}") # RuntimeError 발생
            else: # 로드 경로가 설정되지 않았으면
                print("Error: 'load_model_path' is not specified in the config file for 'finetune' mode.") # 오류 메시지 출력
                print("Please specify the path to the saved model.") # 경로 지정 요청 메시지 출력
                raise ValueError("'load_model_path' must be specified for 'finetune' mode.") # ValueError 발생
        elif model_load_mode == 'pretrained': # 모델 로드 모드가 'pretrained'이면
            print(f"Estimator_v4 model initialized and pretrained weights loaded.") # 사전 학습 가중치 로드 메시지 출력
        else: # 알 수 없는 모델 로드 모드이면
            print(f"Warning: Unknown model_load_mode '{model_load_mode}'. Initializing with pretrained weights.") # 경고 메시지 출력
            print(f"Estimator_v4 model initialized and pretrained weights loaded.") # 사전 학습 가중치 로드 메시지 출력


        self.set_optimizer() # 옵티마이저 설정 메서드 호출

        # Early Stopping 설정
        self._early_stopping = None # early stopping 객체 초기화
        # 설정에서 early stopping 사용 여부 확인
        if self._conf['training'].get('use_early_stopping', False): # early stopping 사용 설정이 True이면
            self._early_stopping = EarlyStopping( # EarlyStopping 객체 생성
                patience=self._conf['training'].get('patience', 7), # patience 설정 (기본값 7)
                verbose=self._conf['training'].get('early_stopping_verbose', False), # 상세 출력 활성화 (설정 파일에서 로드)
                delta=self._conf['training'].get('delta', 0), # delta 설정 (기본값 0)
                path=self._conf['training'].get('checkpoint_path', 'checkpoint.pt') # 체크포인트 경로 설정 (기본값 'checkpoint.pt')
            )

    def set_optimizer(self): # 옵티마이저 설정 메서드
        # 전이 학습을 위해 훈련 가능한 파라미터만 가져와 옵티마이저 설정
        # Estimator_v4에서는 LoRA 파라미터만 훈련 가능하도록 설정되어 있습니다.
        ch_params = [p for n, p in self._estimator.named_parameters() if p.requires_grad] # 훈련 가능한 파라미터 가져오기 (PEFT가 자동으로 설정)
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr']) # Adam 옵티마이저 설정 (LoRA 파라미터에만 적용)

        # 스케줄러 사용 여부 확인
        if self._conf['training'].get('use_scheduler', False):  # 'use_scheduler'가 True일 때만 사용'
            num_warmup_steps = self._conf['training'].get('num_warmup_steps', 0) # warm-up 단계 수 설정 파일에서 로드
            self._ch_scheduler = get_cosine_schedule_with_warmup( # Cosine Annealing with Warmup 스케줄러 생성
                self._ch_optimizer, # 옵티마이저 전달
                num_warmup_steps=num_warmup_steps, # warm-up 단계 수 전달
                num_training_steps=self._conf['training']['num_iter'] # 총 훈련 단계 수 전달
            )
        else: # 스케줄러 사용 설정이 False이면
            self._ch_scheduler = None # 스케줄러 사용 안 함

    def train(self): # 훈련 메서드
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1) # 채널 손실 가중치 (설정 파일에서 로드, 기본값 1)

        # 훈련 데이터로더를 순회하며 훈련
        for it, data in enumerate(self._dataloader): # 훈련 데이터로더 순회 (이터레이션 번호와 데이터 가져오기)
            self._estimator.train() # 모델을 훈련 모드로 설정 (드롭아웃 등 활성화)
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리하여 NumPy 배열 생성
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 지정된 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정 (위상 잡음 추정 결과는 무시)

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기 (복소수 형태)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산 (배치별)
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산 (배치별)
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산 (배치 평균)
            ch_mse = torch.mean(ch_mse) # 채널 MSE 평균 계산 (배치 평균)
            ch_loss = ch_nmse * ch_loss_weight # 채널 손실 계산 (NMSE에 가중치 적용)

            self._ch_optimizer.zero_grad() # 옵티마이저의 그래디언트 초기화
            
            # 역전파 전 학습 가능한 파라미터 확인
            trainable_params = [p for p in self._estimator.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                print(f"ERROR: No trainable parameters found at iteration {it + 1}")
                print("This usually happens after model saving. Checking parameter states...")
                for name, param in self._estimator.named_parameters():
                    if 'lora' in name.lower():
                        print(f"LoRA param {name}: requires_grad={param.requires_grad}")
                break  # 훈련 중단
            
            try:
                ch_loss.backward() # 역전파를 통해 그래디언트 계산
                # LoRA 파라미터만 클리핑
                torch.nn.utils.clip_grad_norm_(self._estimator.parameters(), max_norm=self._conf['training']['max_norm']) # 그래디언트 클리핑 (PEFT 모델의 모든 학습 가능한 파라미터에 적용)
                self._ch_optimizer.step() # 옵티마이저 스텝 (파라미터 업데이트)
            except RuntimeError as e:
                print(f"ERROR during backward pass at iteration {it + 1}: {e}")
                print(f"ch_loss requires_grad: {ch_loss.requires_grad}")
                print(f"ch_loss grad_fn: {ch_loss.grad_fn}")
                print("Checking model parameters:")
                trainable_count = 0
                for name, param in self._estimator.named_parameters():
                    if param.requires_grad:
                        trainable_count += 1
                        print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                print(f"Total trainable parameters: {trainable_count}")
                if trainable_count == 0:
                    print("No trainable parameters found - this is the root cause of the error")
                break  # 훈련 중단

            # 스케줄러 사용 시 학습률 업데이트
            if self._ch_scheduler: # 스케줄러 사용 설정이 True이면
                self._ch_scheduler.step() # 스케줄러 스텝 (학습률 업데이트)

            # 로깅 스텝마다 정보 출력 및 로깅
            if (it + 1) % self._conf['training']['logging_step'] == 0: # 현재 이터레이션이 로깅 스텝의 배수이면
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._conf['training']['lr'] # 현재 학습률 가져오기 (스케줄러 사용 시 스케줄러 학습률, 아니면 초기 학습률)
                print(f"iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}") # 훈련 상태 출력 (이터레이션, 채널 NMSE, 학습률)
                self._logging(it, ch_nmse, ch_est, ch_true) # 로깅 함수 호출

            # 검증 및 Early Stopping 체크
            if (it + 1) % self._conf['training']['evaluation_step'] == 0: # 현재 이터레이션이 평가 스텝의 배수이면
                if self._early_stopping: # early stopping 사용 설정이 True이면
                    val_loss = self.evaluate() # 검증 손실 계산
                    self._early_stopping(val_loss, self._estimator) # early stopping 객체에 검증 손실 전달
                    if self._early_stopping.early_stop: # early stopping 조건 충족 시
                        print("Early stopping") # 메시지 출력
                        break # 훈련 루프 중단


            # 설정된 최대 이터레이션에 도달하면 훈련 중단
            if it >= self._conf['training']['num_iter'] - 1: # 현재 이터레이션이 최대 이터레이션 수에 도달하면
                break # 훈련 루프 중단

        # early stopping으로 훈련이 중단된 경우 최적 모델 로드
        if self._early_stopping and self._early_stopping.early_stop: # early stopping이 활성화되었고 조기 중단 조건이 충족되었으면
             # Early Stopping 시 저장된 병합 모델을 다시 로드 (engine.py와 호환 형태로 저장됨)
             best_model = torch.load(self._early_stopping.path, weights_only=False)
             # 기존 PEFT 모델을 제거하고 새로운 Estimator_v4로 교체
             self._estimator = Estimator_v4(self._conf_file).to(self._device)
             self._estimator.load_state_dict(best_model.state_dict(), strict=False)
             print(f"Loaded best merged model from {self._early_stopping.path}") # 로드 메시지 출력

        # 훈련 완료 후 전체 모델의 state_dict를 .pt 파일로 저장
        self.save_combined_model_as_pt(self._conf['training'].get('saved_model_name', 'final_model')) # 최종 모델을 .pt 파일로 저장

    @torch.no_grad() # 그래디언트 계산 비활성화 (평가 모드)
    def evaluate(self): # 평가 메서드
        if self._val_dataloader is None: # 검증 데이터가 없으면
            print("No validation data available, skipping evaluation.") # 검증 생략 메시지 출력
            return 0.0 # 기본값 반환
        
        self._estimator.eval() # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
        total_nmse = 0.0 # 총 NMSE 초기화
        num_batches = 0 # 배치 카운터 초기화
        # 검증 데이터로더를 순회하며 평가
        for data in self._val_dataloader: # 검증 데이터로더 순회
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리하여 NumPy 배열 생성
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 지정된 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정 (위상 잡음 추정 결과는 무시)

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기 (복소수 형태)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산 (배치별)
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산 (배치별)
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산 (배치 평균)

            total_nmse += ch_nmse.item() # 총 NMSE에 현재 배치의 NMSE 추가
            num_batches += 1 # 배치 카운터 증가

        avg_nmse = total_nmse / num_batches # 평균 NMSE 계산
        print(f"Validation NMSE: {avg_nmse}") # 검증 NMSE 출력
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.log({'val_ch_nmse': avg_nmse}) # 검증 NMSE 로깅
        return avg_nmse # 평균 NMSE 반환


    @torch.no_grad() # 그래디언트 계산 비활성화 (로깅 모드)
    def _logging(self, it, ch_nmse, ch_est, ch_true): # 로깅 메서드
        log = {'ch_nmse': ch_nmse} # 훈련 NMSE 로깅 데이터 딕셔너리 생성
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.log(log) # 훈련 NMSE 로깅
        if (it + 1) % self._conf['training']['evaluation_step'] == 0: # 현재 이터레이션이 평가 스텝의 배수이면 (평가와 동일한 간격으로 플롯 로깅)
            show_batch_size = self._conf['training']['evaluation_batch_size'] # 플롯팅할 배치 크기 설정
            ch_true = ch_true[:, :, 0] + 1j * ch_true[:, :, 1] # 실제 채널 복소수 형태로 변환
            ch_true = ch_true[:show_batch_size].detach().cpu().numpy() # 플롯팅할 실제 채널 데이터 (지정된 배치 크기만큼, detach 후 CPU로 이동 및 NumPy 변환)
            ch_est = ch_est[:, :, 0] + 1j * ch_est[:, :, 1] # 추정 채널 복소수 형태로 변환
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy() # 플롯팅할 추정 채널 데이터 (지정된 배치 크기만큼, detach 후 CPU로 이동 및 NumPy 변환)

            sig_dict = {} # 신호 플롯팅을 위한 딕셔너리 초기화
            sig_dict['ch_est_real'] = {'data': ch_est, 'type': 'real'} # 추정 채널 실수부 데이터 추가
            sig_dict['ch_true_real'] = {'data': ch_true, 'type': 'real'} # 실제 채널 실수부 데이터 추가
            sig_dict['ch_est_imag'] = {'data': ch_est, 'type': 'imag'} # 추정 채널 허수부 데이터 추가
            sig_dict['ch_true_imag'] = {'data': ch_true, 'type': 'imag'} # 실제 채널 허수부 데이터 추가 (ch_imag를 ch_true로 수정)

            f = plot_signal(sig_dict, shape=(3, 2)) # 신호 플롯 생성 (3행 2열 형태)
            f.show() # 플롯 표시
            if self._use_wandb: # WandB 사용 설정이 True이면
                wandb.log({'estimation': wandb.Image(f)}) # 플롯 이미지를 WandB에 로깅
            
            # 모델 저장 간격에 따라 모델 저장
            if (it + 1) % self._conf['training'].get('model_save_step', 100000) == 0: # model_save_step마다 모델 저장
                self.save_combined_model_as_pt(f"{self._conf['training'].get('saved_model_name', 'checkpoint')}_iter_{it + 1}") # 중간 체크포인트를 .pt 파일로 저장
    
    def save_model(self, file_name): # 모델 저장 메서드 (이 메서드는 현재 사용되지 않음)
        path = Path(__file__).parents[0].resolve() / 'saved_model' # 모델 저장 디렉토리 경로 생성
        path.mkdir(parents=True, exist_ok=True) # 모델 저장 디렉토리가 없으면 생성
        # LoRA 가중치만 저장 (이 메서드는 save_combined_model_as_pt로 대체됨)
        # self._estimator.save_pretrained(path / file_name) # PEFT 모델의 LoRA 가중치만 저장
        # print(f"PEFT LoRA weights saved to {path / file_name}") # 모델 저장 경로 출력

    def save_combined_model_as_pt(self, file_name): # 기본 모델과 LoRA 가중치가 결합된 전체 모델의 state_dict를 .pt 파일로 저장하는 메서드
        from model.estimator import Estimator
        
        try:
            print(f"Starting model save process for {file_name}")
            
            # 현재 모델의 학습 가능한 파라미터 상태 확인
            trainable_params = [p for p in self._estimator.parameters() if p.requires_grad]
            print(f"Number of trainable parameters before save: {len(trainable_params)}")
            
            path = Path(__file__).parents[0].resolve() / 'saved_model' # 모델 저장 디렉토리 경로 생성
            path.mkdir(parents=True, exist_ok=True) # 모델 저장 디렉토리가 없으면 생성
            full_path = path / f"{file_name}.pt" # .pt 확장자를 포함한 전체 경로 생성
            
            # LoRA 가중치를 원본 모델에 병합 (원본 모델을 복사해서 사용)
            # 원본 PEFT 모델의 상태를 보존하기 위해 복사본에서 merge_and_unload 수행
            import copy
            estimator_copy = copy.deepcopy(self._estimator)
            merged_v4_model = estimator_copy.merge_and_unload() # LoRA 가중치를 원본 모델에 병합
            
            # 원본 Estimator 구조로 변환하여 engine.py와 완전히 호환되도록 함
            original_estimator = Estimator('config.yaml').to(self._device)
            merged_v4_state_dict = merged_v4_model.state_dict()
            
            # v4 모델의 파라미터를 원본 모델 형태로 변환
            original_state_dict = {}
            for key, value in merged_v4_state_dict.items():
                # v4에서 추가된 projection layer들을 원본 MHA 구조에 맞게 변환
                if any(proj in key for proj in ['mha_q_proj', 'mha_k_proj', 'mha_v_proj']):
                    # v4의 projection layer들을 원본 MultiheadAttention의 개별 proj로 매핑
                    if 'mha_q_proj' in key:
                        if 'weight' in key:
                            new_key = key.replace('mha_q_proj.weight', 'mha.q_proj_weight')
                            original_state_dict[new_key] = value
                        elif 'bias' in key:
                            # q_proj_bias는 원본에서는 in_proj_bias에 포함됨 (나중에 처리)
                            layer_idx = key.split('.')[2]
                            q_bias = merged_v4_state_dict[f'ch_tf._layers.{layer_idx}.mha_q_proj.bias']
                            k_bias = merged_v4_state_dict[f'ch_tf._layers.{layer_idx}.mha_k_proj.bias']
                            v_bias = merged_v4_state_dict[f'ch_tf._layers.{layer_idx}.mha_v_proj.bias']
                            original_state_dict[f'ch_tf._layers.{layer_idx}.mha.in_proj_bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
                    elif 'mha_k_proj' in key and 'weight' in key:
                        new_key = key.replace('mha_k_proj.weight', 'mha.k_proj_weight')
                        original_state_dict[new_key] = value
                    elif 'mha_v_proj' in key and 'weight' in key:
                        new_key = key.replace('mha_v_proj.weight', 'mha.v_proj_weight')
                        original_state_dict[new_key] = value
                    # bias는 q_proj에서 이미 처리했으므로 건너뛰기
                    continue
                else:
                    # 다른 파라미터들은 그대로 복사
                    original_state_dict[key] = value
            
            # 원본 estimator에 변환된 파라미터 로드
            original_estimator.load_state_dict(original_state_dict)
            
            torch.save(original_estimator, full_path) # 원본 구조로 변환된 모델을 .pt 파일로 저장
            print(f"Merged model (compatible with engine.py) saved to {full_path}") # 모델 저장 경로 출력
            
            # 추가: LoRA 가중치만 별도 저장 (디버깅/분석 목적)
            lora_path = path / f"{file_name}_lora_weights.pt"
            self._estimator.save_pretrained(lora_path.parent / f"{file_name}_lora_weights")
            print(f"LoRA weights separately saved to {lora_path.parent / f'{file_name}_lora_weights'}")
            
            # 저장 후 원본 PEFT 모델의 학습 가능한 파라미터 상태 재확인
            trainable_params_after = [p for p in self._estimator.parameters() if p.requires_grad]
            print(f"Number of trainable parameters after save: {len(trainable_params_after)}")
            
            # requires_grad 상태가 변경되었다면 경고 출력
            if len(trainable_params) != len(trainable_params_after):
                print("WARNING: Number of trainable parameters changed after model save!")
                print("This might cause the gradient computation error in the next iteration.")
                
        except Exception as e:
            print(f"Error during model saving: {e}")
            print("Continuing training without saving...")
            import traceback
            traceback.print_exc()


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
        'cp_length': 590,  # cyclic prefix length (ns) # CP 길이 (나노초)
        'max_random_tap_delay_cp_proportion': 0.1,  # random tap delay in proportion of CP length # CP 길이 대비 최대 랜덤 탭 지연 비율
        'rnd_seed': 0, # 랜덤 시드
        'num_workers': 0, # 데이터 로더 워커 수
        'is_phase_noise': False, # 위상 잡음 사용 여부
        'is_channel': True, # 채널 사용 여부
        'is_noise': True # 잡음 사용 여부
    }
    # dataset, dataloader = get_dataset_and_dataloader(params=param_dict) # 데이터셋 및 데이터로더 가져오기 (엔진에서 처리하므로 불필요)
    conf_file = 'config_transfer_v4.yaml' # v4 설정 파일 사용
    engine = TransferLearningEngine(conf_file) # TransferLearningEngine 객체 생성 (파라미터는 설정 파일에서 로드)
    engine.load_model() # 모델 로드 및 초기화
    engine.train() # 훈련 시작