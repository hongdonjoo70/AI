# 감귤(하우스감귤 및 노지감귤) 가격 및 판매량 예측 AI

본 프로젝트는 제주 감귤 경매 데이터와 기상청 기상 관측/예보 데이터, 대체재인 노지감귤(조생귤, 극조생감귤, 온주) 데이터를 결합하여 품목/거래단위(`kind`)별 및 품종(`type`)별 **판매가격(`price`)**과 **판매량(`volume`)**을 추정·예측하는 프로그램입니다.

---

## 1. 주요 모델 및 기법
1. **XGBoost (GPU 가속)**: `tree_method='hist', device='cuda'`, L1 Loss(`reg:absoluteerror`) 기반 WAPE 오차 최소화
2. **LightGBM (분위수 회귀)**: `objective='regression_l1'` 및 분위수 회귀($q=0.1, 0.9$)를 통한 80% 신뢰구간 추정
3. **Temporal Fusion Transformer (TFT, PyTorch GPU + AMP)**: 기상 및 대체재 변수 결합 딥러닝 시계열 예측
4. **Ensemble**: 검증셋 기준 최적 가중 결합 모델
5. **주간 최고일(Top-2 Peak Days) 예측**: 특정 기준일로부터 향후 1주일 내 최고 가격/물량 2개 날짜 예측 및 비율 기반 정확도 산출

---

## 2. 환경 설정 및 패키지 설치

### 1) 가상환경 생성 (권장)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2) PyTorch (GPU / CUDA) 설치
NVIDIA GPU 가속을 활용하려면 CUDA 버전에 맞는 PyTorch를 먼저 설치합니다.
```bash
# CUDA 12.8 기준 예시
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
*(CPU 환경일 경우 일반 `pip install torch` 사용 가능)*

### 3) 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

---

## 3. 데이터 구조 (`data/`)
- `data/house_tangerine.csv`: 하우스감귤 경매 거래 데이터 (`DATE`, `거래단위`, `price`, `volume`, `total price`, `type`, `등급`)
- `data/forecast.csv`: 기상 데이터 (평균기온, 강수 계속시간, 1시간 최다강수량, 합계 일조시간, 적설 등)
- `data/tangerine.csv`: 노지감귤 거래 데이터 (`조생귤`, `극조생감귤`, `온주`)

---

## 4. 실행 방법
Jupyter Lab 또는 VS Code / IDE에서 [`house_tangerine_forecasting.ipynb`](house_tangerine_forecasting.ipynb) 노트북을 열고 실행합니다.
