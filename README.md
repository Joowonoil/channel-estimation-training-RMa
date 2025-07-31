# DNN Channel Estimation Training

> PyTorch 기반 DMRS 채널 추정을 위한 DNN 모델 훈련 시스템

## 🚀 빠른 시작 (Vast AI)

### 🎯 완전 자동 설치 (권장)
```bash
# 새 Vast AI 인스턴스에서 이 한 줄만 실행
curl -sSL https://raw.githubusercontent.com/joowonoil/channel-estimation-training/main/setup_vast_ai.sh | bash

# 설치 완료 후 바로 실행
python Transfer_v4.py
```

### 📋 수동 설치
```bash
# 1. 프로젝트 클론 (Git LFS로 모든 데이터 자동 다운로드)
git clone https://github.com/joowonoil/channel-estimation-training.git
cd channel-estimation-training

# 2. InF LoRA 전이학습 실행
python Transfer_v4.py
```

## 📋 프로젝트 구조

```
channel-estimation-training/
├── 📁 model/                    # DNN 모델 아키텍처
│   ├── estimator.py            # 기본 채널 추정 모델
│   ├── estimator_v3.py         # Adapter 기반 모델
│   ├── estimator_v4.py         # LoRA 기반 모델
│   └── transformer.py          # Transformer 구조
├── 📁 dataset/                 # 데이터셋 (Git LFS)
│   └── PDP_processed/          # 처리된 PDP 데이터
├── 📁 config/                  # 설정 파일들
│   ├── config.yaml             # 기본 설정
│   └── config_transfer_v4.yaml # v4 모델 설정
├── 📁 saved_model/             # 훈련된 모델 (Git LFS)
└── 📁 sample_data*/            # 샘플 데이터 (Git LFS)
```

## 🔧 모델 종류

### 1. Transfer.py - 기본 모델
- 표준 Transformer 기반 채널 추정
- 기본적인 전이학습 지원

### 2. Transfer_v3.py - Adapter 모델  
- Parameter-Efficient Fine-tuning
- Adapter 레이어를 통한 효율적 학습

### 3. Transfer_v4.py - LoRA 모델
- Low-Rank Adaptation (LoRA) 적용
- 최소한의 파라미터로 높은 성능

## 🐳 Docker 환경

### 사전 준비된 환경 사용
```bash
# Vast AI에서 Docker 이미지 사용
docker pull joowonoil/channel-estimation-env:latest
docker run --gpus all -it joowonoil/channel-estimation-env:latest
```

### 포함된 라이브러리
- PyTorch 2.4.1 + CUDA 12.1
- TensorRT, ONNX 최적화 도구
- transformers, peft (LoRA)
- 모든 필요한 의존성 패키지

## ⚡ TensorRT 변환

```bash
# ONNX 및 TensorRT 엔진 생성
python tensorrt_conversion_v4.py
```

## 📊 데이터셋

### 채널 타입 (InF 특화)
- **InF (Indoor Factory)**: Los/Nlos 50000 샘플
- 실내 공장 환경에서의 채널 특성
- LoRA 전이학습을 통한 효율적 모델 적응

### 데이터 형식
- **PDP 파일**: `*.mat` (MATLAB 형식)
- **샘플 데이터**: `*.npy`, `*.npz` (NumPy 형식)
- **모델 파일**: `*.pt` (PyTorch 형식)

## 🛠️ 개발 환경 설정

### 로컬 개발
```bash
# 의존성 설치
pip install -r requirements.txt

# Git LFS 설정 (처음 한 번만)
git lfs install
```

### 설정 파일 수정
```bash
# 모델 설정 변경
vim config/config_transfer_v4.yaml

# 데이터 경로, 배치 크기, 학습률 등 조정 가능
```

## 🔬 실험 관리

### Weights & Biases 연동
```bash
# Wandb 로그인 (선택사항)
wandb login

# 실험 결과는 자동으로 wandb에 기록됨
```

### 모델 저장
- 훈련된 모델: `saved_model/`
- TensorRT 엔진: `tensorrt_model/`
- 체크포인트: 자동 저장

## 📈 성능 최적화

### GPU 메모리 최적화
- 배치 크기 조정: `config/*.yaml`
- Mixed Precision 지원
- Gradient Checkpointing

### 추론 가속화
- TensorRT 엔진 변환
- ONNX 최적화
- 다중 GPU 지원

## 🤝 기여 방법

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 🙋‍♂️ 지원

문제가 발생하면 [Issues](https://github.com/joowonoil/channel-estimation-training/issues)에 등록해주세요.

---

**🎯 목표**: InF 채널 환경에서 LoRA 전이학습을 통한 DMRS 기반 5G/6G 채널 추정 성능 향상