#!/bin/bash

# Vast AI에서 실행할 완전한 설정 스크립트

echo "=== Vast AI Channel Estimation Setup ==="

# 1. 환경 확인
echo "1. Checking environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
nvidia-smi

# 2. 프로젝트 클론
echo "2. Cloning project..."
git clone https://github.com/joowonoil/channel-estimation-training.git
cd channel-estimation-training

# 3. 데이터셋 디렉토리 생성 (필요시)
echo "3. Creating dataset directories..."
mkdir -p dataset/PDP_processed
mkdir -p sample_data
mkdir -p saved_model

# 4. 환경 테스트
echo "4. Testing imports..."
python -c "
try:
    import torch
    import numpy as np
    import yaml
    import wandb
    import einops
    import transformers
    import peft
    print('✅ All imports successful!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# 5. GPU 메모리 확인
echo "5. GPU memory check..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
else:
    print('No GPU available')
"

echo "=== Setup Complete! ==="
echo "Ready to run training scripts:"
echo "  python Transfer.py"
echo "  python Transfer_v3.py" 
echo "  python Transfer_v4.py"