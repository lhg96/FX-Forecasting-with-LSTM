#!/bin/bash

# 환율 예측 LSTM 프로젝트 설정 스크립트

echo "======================================"
echo "환율 예측 LSTM 프로젝트 설정"
echo "======================================"

# 1. 가상환경 확인
if [ -d "venv" ]; then
    echo "✓ 가상환경이 이미 존재합니다."
else
    echo "가상환경 생성 중..."
    python -m venv venv
    echo "✓ 가상환경 생성 완료"
fi

# 2. 가상환경 활성화
echo ""
echo "가상환경 활성화 중..."
source venv/bin/activate

# 3. pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip

# 4. 패키지 설치
echo ""
echo "필요한 패키지 설치 중..."
pip install -r requirements.txt

# 5. 완료
echo ""
echo "======================================"
echo "설정 완료!"
echo "======================================"
echo ""
echo "다음 명령어로 가상환경을 활성화하세요:"
echo "  source venv/bin/activate"
echo ""
echo "프로젝트 실행:"
echo "  cd src"
echo "  python main.py"
echo ""
