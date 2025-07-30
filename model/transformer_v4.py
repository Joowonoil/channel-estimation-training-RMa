import numpy as np # NumPy 라이브러리 임포트 (수치 연산용)
import math # 수학 함수 임포트
import torch # PyTorch 라이브러리 임포트 (딥러닝 프레임워크)
import torch.nn.functional as F # PyTorch의 함수형 API 임포트
from torch import nn, Tensor # 신경망 모듈 및 텐서 타입 임포트
#from einops import rearrange, repeat # einops 라이브러리 임포트 (텐서 재배열용, 현재 주석 처리됨)
from model.positional_encoding import positional_encoding_sine # 위치 인코딩 함수 임포트
from model.multi_head_attention import MultiheadAttention # MultiheadAttention 모듈 임포트
from torch.nn import Linear, LayerNorm, Dropout # Linear, LayerNorm, Dropout 모듈 임포트
from torch.nn.parameter import Parameter # 모델 파라미터 정의를 위한 Parameter 임포트
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, normal_ # 파라미터 초기화 함수 임포트


class ConditionNetwork(nn.Module): # ConditionNetwork 클래스 정의 (입력 데이터를 트랜스포머에 맞게 변환)
    def __init__(self, length, in_channels, step_size, steps_per_token): # 초기화 메서드
        super(ConditionNetwork, self).__init__() # 부모 클래스 초기화
        self._length = length # 입력 데이터 길이 저장
        self._in_channels = in_channels # 입력 채널 수 저장
        self._step_size = step_size # 스텝 크기 저장
        self._steps_per_token = steps_per_token # 토큰 당 스텝 수 저장
        self._d_model = step_size * steps_per_token * in_channels # 모델 차원 계산
        if self._length % self._step_size != 0: # 데이터 길이가 스텝 크기로 나누어 떨어지는지 확인
            raise Exception('Length is not divisible by step size') # 나누어 떨어지지 않으면 예외 발생
        self._n_token = int(self._length / self._step_size) - steps_per_token + 1 # 토큰 수 계산

    @property # n_token 속성 정의
    def n_token(self):
        return self._n_token # 토큰 수 반환

    @property # d_model 속성 정의
    def d_model(self):
        return self._d_model # 모델 차원 반환

    def forward(self, c): # 순전파 메서드
        # n_batch, n_channel, n_data = c.shape # 입력 텐서의 배치, 채널, 데이터 길이 가져오기 (주석 처리됨)
        # if n_channel != self._in_channels: # 입력 채널 수가 설정과 일치하는지 확인 (주석 처리됨)
        #     raise Exception('Channel does not match') # 일치하지 않으면 예외 발생 (주석 처리됨)
        # if n_data != self._length: # 입력 데이터 길이가 설정과 일치하는지 확인 (주석 처리됨)
        #     raise Exception('Data length does not match') # 일치하지 않으면 예외 발생 (주석 처리됨)
        # c = c.unfold(dimension=-1, size=self._step_size * self._steps_per_token, step=self._step_size) # unfold 연산 (주석 처리됨)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1, self._step_size)) # 입력 텐서 형태 변경
        # c = rearrange(c, 'b c t d -> t b (c d)') # rearrange 연산 (주석 처리됨)
        c = c.permute(2, 0, 1, 3) # 텐서 차원 재배열
        c = c.flatten(start_dim=2, end_dim=3) # 텐서 평탄화
        return c # 변환된 텐서 반환


class Transformer(nn.Module): # Transformer 모델 클래스 정의
    def __init__(self, length, channels, num_layers, d_model, n_token, n_head, dim_feedforward, dropout, # 초기화 메서드
                 activation="relu", cond_net=None): # LoRA 관련 파라미터 제거
        super(Transformer, self).__init__() # 부모 클래스 초기화
        self._length = length # 입력 데이터 길이 저장
        self._channels = channels # 입력 채널 수 저장
        self._num_layers = num_layers # 트랜스포머 레이어 수 저장
        self._d_model = d_model # 모델 차원 저장
        self._n_token = n_token # 토큰 수 저장
        if self._length % self._n_token != 0: # 데이터 길이가 토큰 수로 나누어 떨어지는지 확인
            raise Exception('Length is not divisible by number of tokens') # 나누어 떨어지지 않으면 예외 발생
        self._step_size = int((self._channels * self._length) / self._n_token) # 스텝 크기 계산
        self._n_head = n_head # 어텐션 헤드 수 저장
        self._dim_feedforward = dim_feedforward # 피드포워드 네트워크 차원 저장
        self._dropout = dropout # 드롭아웃 비율 저장
        self._activation = activation # 활성화 함수 저장
        self._cond_net = cond_net # ConditionNetwork 인스턴스 저장
        self._cond_d_model = cond_net.d_model # ConditionNetwork의 모델 차원 저장
        self._cond_n_token = cond_net.n_token # ConditionNetwork의 토큰 수 저장

        # Transformer layers # 트랜스포머 레이어 정의
        self._layers = nn.ModuleList() # ModuleList 생성
        for _ in range(self._num_layers): # 설정된 레이어 수만큼 반복
            layer = TransformerLayer(self._d_model, self._n_token, n_head, dim_feedforward, dropout, # TransformerLayer 인스턴스 생성
                                     activation="relu", cond_d_model=self._cond_d_model, # 활성화 함수, ConditionNetwork 차원 및 토큰 수 전달
                                     cond_n_token=self._cond_n_token,
                                    )
            self._layers.append(layer) # 생성된 레이어를 ModuleList에 추가

        self._embedding = Parameter(torch.empty((self._n_token, self._d_model), requires_grad=True)) # 임베딩 파라미터 정의 및 초기화
        self._linear = Linear(in_features=self._d_model, out_features=self._step_size, bias=True) # 최종 Linear 레이어 정의
        self._reset_parameters() # 파라미터 초기화 메서드 호출

    def _reset_parameters(self): # 파라미터 초기화 메서드
        nn.init.normal_(self._embedding) # 임베딩 파라미터 정규 분포 초기화
        xavier_uniform_(self._linear.weight) # Linear 레이어 가중치 Xavier 균등 분포 초기화
        constant_(self._linear.bias, 0.) # Linear 레이어 bias 0으로 초기화

    def forward(self, cond): # 순전파 메서드
        c = self._cond_net(cond) # ConditionNetwork를 통해 입력 데이터 변환
        n_batch = c.shape[1] # 배치 크기 가져오기
        #x = repeat(self._embedding, 't d -> t b d', b=n_batch) # 임베딩 반복 (주석 처리됨)
        x = self._embedding[:, None, :].repeat(1, n_batch, 1) # 임베딩을 배치 크기만큼 반복
        for i, layer in enumerate(self._layers): # 각 트랜스포머 레이어를 순회
            x = layer(input=x, cond=c) # 트랜스포머 레이어 순전파
        #y = rearrange(self._linear(x), 't b (c d) -> b c (t d)', c=self._channels) # rearrange 연산 (주석 처리됨)
        y = self._linear(x) # 최종 Linear 레이어 순전파
        y = torch.reshape(y, shape=(y.shape[0], y.shape[1], self._channels, -1)) # 출력 텐서 형태 변경
        y = y.permute(1, 2, 0, 3) # 텐서 차원 재배열
        y = y.flatten(start_dim=2, end_dim=3) # 텐서 평탄화
        return y # 최종 결과 반환


class TransformerLayer(nn.Module): # TransformerLayer 클래스 정의 (트랜스포머의 단일 레이어)
    def __init__(self, d_model, n_token, n_head, dim_feedforward, dropout, activation="relu", # 초기화 메서드
                 cond_d_model=None, cond_n_token=None, lora_rank=None, lora_alpha=None): # LoRA 관련 파라미터 추가
        super(TransformerLayer, self).__init__() # 부모 클래스 초기화
        self._d_model = d_model # 모델 차원 저장
        self._n_token = n_token # 토큰 수 저장
        self._n_head = n_head # 어텐션 헤드 수 저장
        self._dim_feedforward = dim_feedforward # 피드포워드 네트워크 차원 저장
        self._dropout = dropout # 드롭아웃 비율 저장
        self._activation = activation # 활성화 함수 저장
        self._cond_d_model = cond_d_model # ConditionNetwork의 모델 차원 저장
        self._cond_n_token = cond_n_token # ConditionNetwork의 토큰 수 저장

        # Positional encoding and mask # 위치 인코딩 및 마스크
        pe = positional_encoding_sine(n_pos=n_token, d_model=d_model, max_n_pos=n_token, # 위치 인코딩 생성
                                      normalize=True, scale=None)
        self.register_buffer('pe', pe) # 위치 인코딩 버퍼 등록
        cond_pe = positional_encoding_sine(n_pos=cond_n_token, d_model=cond_d_model, max_n_pos=cond_n_token, # ConditionNetwork 위치 인코딩 생성
                                           normalize=True, scale=None)
        self.register_buffer('cond_pe', cond_pe) # ConditionNetwork 위치 인코딩 버퍼 등록
        # Multihead attention modules # Multihead attention 모듈
        # Self-attention projection layers
        self.mha_q_proj = Linear(d_model, d_model, bias=True)
        self.mha_k_proj = Linear(d_model, d_model, bias=True)
        self.mha_v_proj = Linear(d_model, d_model, bias=True)
        self.mha = MultiheadAttention(embed_dim=d_model, num_heads=n_head, kdim=d_model, vdim=d_model, # Self-attention 모듈 정의
                                      dropout=dropout, bias=True, dtype=None)
        
        # Cross-attention (원본 방식과 동일하게 수정 - projection 없이 직접 사용)
        self.cond_mha = MultiheadAttention(embed_dim=d_model, num_heads=n_head, kdim=cond_d_model, vdim=cond_d_model, # Cross-attention 모듈 정의 (원본과 동일)
                                           dropout=dropout, bias=True,
                                           dtype=None)
        # Feedforward neural network # 피드포워드 신경망
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True) # 첫 번째 Linear 레이어 정의
        self.ffnn_dropout = Dropout(dropout) # 피드포워드 드롭아웃 정의
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True) # 두 번째 Linear 레이어 정의
        self.activation = _get_activation_fn(activation) # 활성화 함수 가져오기
        # Layer norm and dropout # Layer norm 및 드롭아웃
        layer_norm_eps = 1e-5 # Layer norm epsilon 값
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps) # 첫 번째 Layer norm 정의
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps) # 두 번째 Layer norm 정의
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps) # 세 번째 Layer norm 정의
        self.dropout1 = Dropout(dropout) # 첫 번째 드롭아웃 정의
        self.dropout2 = Dropout(dropout) # 두 번째 드롭아웃 정의
        self.dropout3 = Dropout(dropout) # 세 번째 드롭아웃 정의

        # Reset parameters # 파라미터 초기화
        self._reset_parameters() # 파라미터 초기화 메서드 호출

    def _reset_parameters(self): # 파라미터 초기화 메서드
        xavier_uniform_(self.mha_q_proj.weight)
        constant_(self.mha_q_proj.bias, 0.)
        xavier_uniform_(self.mha_k_proj.weight)
        constant_(self.mha_k_proj.bias, 0.)
        xavier_uniform_(self.mha_v_proj.weight)
        constant_(self.mha_v_proj.bias, 0.)

        # cond_mha projection 제거됨 (원본 방식 사용)

        xavier_uniform_(self.ffnn_linear1.weight) # ffnn_linear1 가중치 Xavier 균등 분포 초기화
        xavier_uniform_(self.ffnn_linear2.weight) # ffnn_linear2 가중치 Xavier 균등 분포 초기화
        constant_(self.ffnn_linear1.bias, 0.) # ffnn_linear1 bias 0으로 초기화
        constant_(self.ffnn_linear2.bias, 0.) # ffnn_linear2 bias 0으로 초기화

    def forward(self, input, cond=None): # 순전파 메서드
        # Pre-normalization # Pre-normalization 방식 적용
        # Self-attention # Self-attention 블록
        norm_x = self.norm1(input) # 첫 번째 Layer norm 적용
        q = self.mha_q_proj(norm_x)
        k = self.mha_k_proj(norm_x)
        v = self.mha_v_proj(norm_x)
        x2, _ = self.mha(query=q, key=k, value=v, query_pos=self.pe, key_pos=self.pe, # Self-attention 계산
                         attn_mask=None, need_weights=False, average_attn_weights=False)
        x = input + self.dropout1(x2) # 잔차 연결 및 드롭아웃 적용

        # Cross-attention # Cross-attention 블록
        norm_x = self.norm2(x) # 두 번째 Layer norm 적용
        q_cond = self.mha_q_proj(norm_x) # query는 self-attention과 동일한 projection 사용
        x2, _ = self.cond_mha(query=q_cond, key=cond, value=cond, query_pos=self.pe, key_pos=self.cond_pe, # Cross-attention 계산 (원본 방식)
                              attn_mask=None, need_weights=False, average_attn_weights=False)
        x = x + self.dropout2(x2) # 잔차 연결 및 드롭아웃 적용

        # Feedforward # Feedforward 블록
        norm_x = self.norm3(x) # 세 번째 Layer norm 적용
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(norm_x)))) # Feedforward 네트워크 순전파
        x = x + self.dropout3(x2) # 잔차 연결 및 드롭아웃 적용
        return x # 최종 결과 반환


def _get_activation_fn(activation): # 활성화 함수를 가져오는 헬퍼 함수
    """Return an activation function given a string""" # 문자열에 해당하는 활성화 함수 반환
    if activation == "relu": # "relu"이면
        return F.relu # ReLU 함수 반환
    if activation == "gelu": # "gelu"이면
        return F.gelu # GELU 함수 반환
    if activation == "glu": # "glu"이면
        return F.glu # GLU 함수 반환
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.") # 지원하지 않는 활성화 함수이면 예외 발생