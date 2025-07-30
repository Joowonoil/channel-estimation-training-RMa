# 🧠 DNN Channel Estimation - Notion 문서

## 📋 프로젝트 개요

### 🎯 목표
- **DMRS 기반 5G/6G 채널 추정**의 정확도와 효율성 향상
- **PyTorch + Transformer** 아키텍처 활용
- **클라우드 환경 (Vast AI)**에서 대용량 모델 훈련

### 🛠️ 기술 스택
| 카테고리 | 기술 |
|---------|------|
| **프레임워크** | PyTorch 2.4.1, CUDA 12.1 |
| **모델** | Transformer, LoRA, Adapter |
| **최적화** | TensorRT, ONNX |
| **클라우드** | Vast AI, Docker |
| **버전관리** | Git + Git LFS |

---

## 🚀 빠른 시작 가이드

### Step 1: 환경 준비
```bash
# Vast AI 템플릿 사용
Docker Image: joowonoil/channel-estimation-env:latest
GPU: RTX 4090 (CUDA 12.1 호환)
Disk: 20GB+
```

### Step 2: 프로젝트 클론
```bash
git clone https://github.com/joowonoil/channel-estimation-training.git
cd channel-estimation-training
```

### Step 3: 모델 실행
```bash
# 최신 LoRA 모델 (추천)
python Transfer_v4.py

# 기본 모델
python Transfer.py

# Adapter 모델  
python Transfer_v3.py
```

---

## 📁 프로젝트 구조 (변동 가능)

### 핵심 디렉토리
```
📦 프로젝트 루트
├── 🎯 model/              # 신경망 아키텍처
├── 📊 dataset/            # 훈련 데이터 (Git LFS)
├── ⚙️ config/             # 설정 파일들
├── 💾 saved_model/        # 훈련된 모델 저장
├── 🔧 utils/              # 유틸리티 함수들
└── 📋 requirements.txt    # Python 의존성
```

### 주요 파일 패턴
- **훈련 스크립트**: `Transfer*.py`
- **모델 정의**: `model/estimator*.py`
- **설정 파일**: `config/config*.yaml`
- **변환 도구**: `tensorrt_conversion*.py`

---

## 🔧 설정 및 커스터마이징

### 모델 설정 변경
**위치**: `config/config_transfer_v4.yaml` (또는 해당 버전)

```yaml
# 주요 설정 항목들 (실제 파일 확인 필요)
model:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100

data:
  channel_type: ["InF_Los", "RMa_Los"]  # 필요에 따라 변경
  data_path: "./dataset/"
```

### 하이퍼파라미터 튜닝 포인트
1. **학습률 (Learning Rate)**: 모델 수렴 속도 조절
2. **배치 크기 (Batch Size)**: GPU 메모리와 성능 균형
3. **채널 타입**: 훈련할 채널 환경 선택
4. **모델 크기**: Transformer 레이어 수, 헤드 수

---

## 🎛️ 모델 버전별 특징

### 🔹 Transfer.py (기본)
- **용도**: 표준 Transformer 기반 학습
- **특징**: 안정적, 이해하기 쉬움
- **적합**: 초기 실험, 베이스라인

### 🔹 Transfer_v3.py (Adapter)
- **용도**: Parameter-Efficient Fine-tuning
- **특징**: 적은 파라미터로 효율적 학습
- **적합**: 제한된 컴퓨팅 환경

### 🔹 Transfer_v4.py (LoRA)
- **용도**: Low-Rank Adaptation
- **특징**: 최신 기법, 높은 성능
- **적합**: 최적 성능 추구

---

## 🐳 Docker 환경 관리

### 사전 구축된 환경
```bash
# 환경 이미지 pull
docker pull joowonoil/channel-estimation-env:latest

# 로컬 테스트
docker run --gpus all -it joowonoil/channel-estimation-env:latest
```

### 환경 포함 내용
- ✅ PyTorch 2.4.1 + CUDA 12.1
- ✅ 모든 Python 의존성 패키지
- ✅ TensorRT, ONNX 최적화 도구
- ✅ 개발에 필요한 시스템 도구

---

## 📊 데이터 관리 (Git LFS)

### 지원되는 파일 형식
| 확장자 | 용도 | Git LFS |
|--------|------|---------|
| `*.mat` | MATLAB 데이터 | ✅ |
| `*.npy`, `*.npz` | NumPy 배열 | ✅ |
| `*.pt`, `*.pth` | PyTorch 모델 | ✅ |
| `*.engine` | TensorRT 엔진 | ✅ |

### 채널 타입 (변경 가능)
- **InF**: Indoor Factory (Los/NLos)
- **InH**: Indoor Hotspot (Los/NLos)
- **RMa**: Rural Macro (Los/NLos)
- **UMa**: Urban Macro (Los/NLos)
- **UMi**: Urban Micro (Los/NLos)

---

## ⚡ 성능 최적화

### GPU 메모리 최적화
```python
# config 파일에서 조정 가능한 항목들
batch_size: 16          # GPU 메모리에 맞게 조정
gradient_checkpointing: true
mixed_precision: true
```

### TensorRT 가속화
```bash
# 훈련 완료 후 추론 가속화
python tensorrt_conversion_v4.py
```

---

## 🔄 개발 워크플로우

### 1. 로컬 개발
```bash
# 코드 수정
vim Transfer_v4.py
vim config/config_transfer_v4.yaml

# Git 커밋
git add .
git commit -m "Update model architecture"
git push
```

### 2. 클라우드 테스트
```bash
# Vast AI에서
git pull  # 최신 코드 가져오기
python Transfer_v4.py  # 실행
```

### 3. 결과 저장
```bash
# 중요한 모델만 커밋
git add saved_model/best_model.pt
git commit -m "Add trained model - accuracy: 95%"
git push
```

---

## 🛠️ 트러블슈팅

### 일반적인 문제들

#### CUDA 메모리 부족
```bash
# 해결방법
1. config에서 batch_size 줄이기
2. 모델 크기 축소
3. gradient_checkpointing 활성화
```

#### 데이터 로딩 오류
```bash
# 확인사항
1. dataset/ 폴더 존재 확인
2. .mat 파일들 정상 다운로드 확인
3. Git LFS 제대로 설치되었는지 확인
```

#### TensorRT 호환성 문제
```bash
# 해결방법
1. 더 최신 GPU 인스턴스 선택
2. CUDA 버전 호환성 확인
3. torch_tensorrt import 오류 시 해당 코드 주석 처리
```

---

## 📈 실험 관리

### Weights & Biases 연동
```bash
# 선택사항: 실험 추적
export WANDB_API_KEY=your_key
wandb login
```

### 모델 버전 관리
```bash
# 의미있는 커밋 메시지 사용
git commit -m "v4.1: Add attention dropout, lr=0.001, acc=96.5%"
```

---

## 🔮 향후 확장 가능성

### 새로운 모델 추가
1. `model/` 폴더에 새 모델 클래스 작성
2. `Transfer_v5.py` 같은 새 훈련 스크립트 작성
3. `config/config_v5.yaml` 설정 파일 생성

### 새로운 데이터셋 지원
1. `dataset.py`에서 데이터 로더 확장
2. Git LFS에 새 데이터 형식 추가
3. config에서 새 채널 타입 정의

### 배포 환경 확장
1. 다른 클라우드 플랫폼 지원
2. 온프레미스 환경 Docker 설정
3. 모바일/엣지 디바이스 최적화

---

## 🎯 핵심 메트릭

### 성능 지표
- **정확도 (Accuracy)**: 채널 추정 정확도
- **MSE/RMSE**: 평균 제곱 오차
- **추론 시간**: TensorRT 최적화 후 속도
- **메모리 사용량**: GPU VRAM 효율성

### 비즈니스 임팩트
- **5G/6G 통신 품질** 향상
- **에너지 효율성** 개선
- **실시간 처리** 가능성

---

*📝 이 문서는 프로젝트 진화에 맞춰 지속적으로 업데이트됩니다.*