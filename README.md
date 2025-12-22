# Finance Projects - Exchange Rate Forecasting with LSTM 💹

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-green.svg)](https://gradio.app/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Custom-red.svg)](#-license)

This repository contains exchange rate forecasting projects using LSTM neural networks and Bank of Korea API for data collection.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project 1: FX-Forecasting-with-LSTM](#-project-1-fx-forecasting-with-lstm)
- [Project 2: fx-lstm-analysis](#-project-2-fx-lstm-analysis)
- [Project 3: bok-exchange-rate](#-project-3-bok-exchange-rate-new)
- [Project Comparison](#-project-comparison)
- [Learning Resources](#-learning-resources)
- [Development Environment](#️-development-environment)
- [Disclaimer](#️-disclaimer)
- [License](#-license)
- [Contact & Services](#-contact--services)

## 📊 Project Overview

This repository includes three exchange rate related projects:

1. **[FX-Forecasting-with-LSTM](FX-Forecasting-with-LSTM/)**: Gradio web interface-based forecasting system
2. **[fx-lstm-analysis](fx-lstm-analysis/)**: Modular data collection and analysis system
3. **[bok-exchange-rate](bok-exchange-rate/)**: Bank of Korea API exchange rate data collector 🆕

---

## 📁 Project 1: FX-Forecasting-with-LSTM

### Features
- 🌐 **Gradio Web Interface**: Intuitive UI accessible directly from browser
- 📈 **Real-time Data Collection**: Automatic web scraping from financial websites
- 🔄 **Interactive Model Training**: Train and predict directly from the web interface
- 📊 **Visualization**: View prediction results through interactive graphs

### Screenshots

![FX Forecasting Web Interface](FX-Forecasting-with-LSTM/screenshots/main.png)
*Gradio-based web interface*

### Tech Stack
- Python 3.8+
- TensorFlow 2.0+
- Gradio 3.0+
- Pandas, NumPy, Matplotlib
- BeautifulSoup (Web Scraping)

### Key Features
1. **Automated Data Collection**
   - USD/KRW Exchange Rate
   - Dollar Index (DXY)
   - CRB Index

2. **LSTM Model**
   - Time series data learning
   - Adjustable hyperparameters
   - Early stopping capability

3. **Web Interface**
   - Real-time model training
   - Prediction result visualization
   - User-friendly UI

### Getting Started

```bash
cd FX-Forecasting-with-LSTM
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

The Gradio interface will automatically open in your web browser.

### Project Structure
```
FX-Forecasting-with-LSTM/
├── src/
│   ├── main.py              # Gradio app main file
│   ├── data_collector.py    # Data collection module
│   ├── model.py             # LSTM model
│   └── preprocessor.py      # Data preprocessing
├── screenshots/             # Screenshots
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

---

## 📁 Project 2: fx-lstm-analysis

### Features
- 🔧 **Modular Architecture**: Reusable independent modules
- 📊 **Comprehensive Analysis**: Complete pipeline from data collection to prediction
- 📈 **Rich Visualization**: 7+ analysis graphs
- 🎯 **High Accuracy**: R² Score 0.99+ achieved

### Screenshots

#### Model Training Process
![Training Loss](fx-lstm-analysis/screenshots/training_history.png)
*Model training loss graph - Training/validation loss changes per epoch*

#### Prediction Results
![Full Prediction Results](fx-lstm-analysis/screenshots/predictions_full.png)
*Complete prediction results comparison on validation data*

![Recent Predictions](fx-lstm-analysis/screenshots/recent_predictions.png)
*Detailed predictions and error analysis for recent 20 days*

#### Analysis Visualization
![Error Analysis](fx-lstm-analysis/screenshots/error_analysis.png)
*Prediction error distribution and time series analysis*

![Full Data](fx-lstm-analysis/screenshots/full_data_predictions.png)
*Complete data and predictions from 1998-2024*

![Moving Averages](fx-lstm-analysis/screenshots/moving_averages.png)
*Moving average-based trend analysis*

### Tech Stack
- Python 3.12
- TensorFlow 2.20
- Pandas 2.3+
- Scikit-learn 1.8
- Matplotlib 3.10
- OpenPyXL (Excel processing)

### Key Features

1. **Data Collection Module** (`data_collector.py`)
   - Automated web scraping
   - Data merging and storage
   - Existing data updates

2. **Data Preprocessing** (`data_preprocessor.py`)
   - Missing value handling
   - Data normalization
   - Train/validation split

3. **LSTM Model** (`lstm_model.py`)
   - 200-unit LSTM
   - Early stopping (Patience: 40)
   - Checkpoint saving

4. **Model Evaluation** (`model_evaluator.py`)
   - MAE, R² Score calculation
   - Error analysis
   - Trend prediction

5. **Visualization** (`visualizer.py`)
   - Training history
   - Prediction results
   - Error analysis
   - Moving averages

### Performance Metrics

```
MAE (Mean Absolute Error): 8.75 KRW
R² Score: 0.99251
Prediction Accuracy:
  - Within 50 KRW: 99.85%
  - Within 100 KRW: 100.00%
```

### Getting Started

```bash
cd fx-lstm-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt

# Generate sample data (instead of web scraping)
cd src
python generate_sample_data.py

# Run complete pipeline
python main.py
```

Or use automated setup script:
```bash
./setup.sh
cd src
python main.py
```

### Project Structure
```
fx-lstm-analysis/
├── data/
│   └── forex_data.xlsx      # Exchange rate data
├── models/
│   └── best_model.weights.h5 # Trained model
├── output/
│   ├── predictions.xlsx     # Prediction results
│   └── *.png               # Visualization graphs
├── src/
│   ├── main.py             # Main execution script
│   ├── data_collector.py   # Data collection
│   ├── data_preprocessor.py # Preprocessing
│   ├── lstm_model.py       # Model definition
│   ├── model_evaluator.py  # Evaluation
│   ├── visualizer.py       # Visualization
│   └── generate_sample_data.py # Sample data generation
├── screenshots/            # Result screenshots
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

---

## 📁 Project 3: bok-exchange-rate 🆕

### Overview
Official Bank of Korea (BOK) ECOS API를 활용한 환율 데이터 수집 도구입니다. USD, JPY, CNY 등 주요 통화의 환율 정보를 수집하고 분석할 수 있습니다.

### Features
- ✅ **실시간 환율 조회**: 최신 환율 정보 실시간 조회
- ✅ **과거 데이터 수집**: 지정 기간의 환율 데이터 수집
- ✅ **여러 통화 지원**: USD, JPY, CNY 동시 조회 가능
- ✅ **통계 분석**: 최저/최고/평균/변동성 자동 계산
- ✅ **CSV 내보내기**: 데이터 저장 및 공유
- ✅ **완전한 테스트**: Mock 및 통합 테스트 포함
- ✅ **.env 파일 지원**: API 키 안전 관리

### 🚀 Quick Start

```bash
cd bok-exchange-rate

# 패키지 설치
pip install -r requirements_exchange_rate.txt

# .env 파일 생성 (예제 파일 복사)
cp .env.example .env
# .env 파일을 열어서 API 키 입력

# 예제 실행
python example_usage.py
```

### Tech Stack
- Python 3.8+
- Requests 2.31+
- Pandas 2.0+
- python-dotenv 1.0+ (환경변수 관리)

### 📊 Latest Test Results (2025-12-22)

#### 현재 환율 정보
```
미국 달러 (USD): 1,477.80원
일본 엔화 (JPY):   937.78원
중국 위안 (CNY):   210.11원
```

#### USD/KRW 최근 3개월 통계 (2025-09-23 ~ 2025-12-22)
- 데이터 건수: 60건
- 최저 환율: 1,393.80원
- 최고 환율: 1,478.60원
- 평균 환율: 1,445.36원
- 표준 편차: 26.57원
- 최대 상승: 18.60원/일
- 최대 하락: -11.00원/일
- 평균 변동: 3.80원/일

#### 통화별 환율 비교 (최근 6개월)
| 통화 | 최저 | 최고 | 평균 | 변동폭 |
|------|------|------|------|--------|
| USD | 1,352.60원 | 1,478.60원 | 1,413.07원 | 126.00원 |
| JPY | 915.08원 | 956.43원 | 940.37원 | 41.35원 |
| CNY | 189.02원 | 210.24원 | 198.16원 | 21.22원 |

#### 통화간 상관관계 분석
```
          USD       JPY       CNY
USD  1.000000  0.247988  0.993562
JPY  0.247988  1.000000  0.256494
CNY  0.993562  0.256494  1.000000
```
- **USD ↔ CNY**: 0.99 (매우 강한 양의 상관관계)
- **USD ↔ JPY**: 0.25 (약한 양의 상관관계)
- **JPY ↔ CNY**: 0.26 (약한 양의 상관관계)

#### 데이터 수집 성과
- ✅ 2024년 전체 데이터 수집 완료 (245건)
- ✅ CSV 파일 4개 생성
  - `exchange_rate_USD_2024.csv`
  - `exchange_rate_JPY_2024.csv`
  - `exchange_rate_CNY_2024.csv`
  - `exchange_rates_all_2024.csv` (통합)

### 사용 예제

```python
from exchange_rate_fetcher import ExchangeRateFetcher
from dotenv import load_dotenv
import os

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv('BOK_API_KEY')

# API 초기화
fetcher = ExchangeRateFetcher(api_key)

# 최신 환율 조회
latest = fetcher.get_latest_rate('USD')
print(f"현재 USD 환율: {latest['rate']:,.2f}원")

# 과거 데이터 조회
df = fetcher.fetch_exchange_rate('USD', '20240101', '20241231')
print(f"평균 환율: {df['DATA_VALUE'].mean():,.2f}원")
```

### API 키 설정 (.env 파일 사용)

**✅ 권장 방법: .env 파일 사용**

1. [한국은행 ECOS](https://ecos.bok.or.kr/) 가입
2. API 인증키 신청
3. `.env` 파일 생성:
   ```bash
   cp .env.example .env
   ```
4. `.env` 파일 편집:
   ```
   BOK_API_KEY=your_api_key_here
   ```
5. Git에 자동 제외됨 (`.gitignore`에 이미 추가됨)

**대체 방법: 환경변수 직접 설정**
```bash
export BOK_API_KEY='your_api_key_here'
```

### 보안 관리
- ✅ `.env` 파일은 `.gitignore`에 포함되어 Git에 업로드되지 않음
- ✅ `.env.example` 파일 제공으로 설정 방법 안내
- ✅ `python-dotenv` 패키지로 환경변수 안전 관리

### Project Structure
```
bok-exchange-rate/
├── .env                          # API 키 (Git 제외)
├── .env.example                  # 환경변수 샘플
├── .gitignore                    # Git 제외 파일
├── exchange_rate_fetcher.py      # 메인 모듈
├── test_exchange_rate.py         # 테스트 코드
├── example_usage.py              # 사용 예제
├── requirements_exchange_rate.txt # 의존성
├── README.md                     # Quick Start 가이드
└── README_exchange_rate.md       # 상세 API 문서
```

### 참고 자료
- [프로젝트 상세 문서](bok-exchange-rate/README.md)
- [API 문서](bok-exchange-rate/README_exchange_rate.md)
- [참고 블로그](https://yenpa.tistory.com/106)

---


## 📚 Learning Resources

### LSTM Neural Networks
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [TensorFlow LSTM Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)

### Time Series Forecasting
- [Time Series Forecasting with Python](https://machinelearningmastery.com/time-series-forecasting-python-mini-course/)
- [Financial Time Series Analysis](https://www.investopedia.com/terms/t/timeseries.asp)

---

## 🛠️ Development Environment

- **Python**: 3.8+ (Project 1), 3.12 (Project 2)
- **TensorFlow**: 2.0+
- **OS**: macOS, Linux, Windows

---

## ⚠️ Disclaimer

**Important**: The predictions from these models are NOT investment advice.

- Predictions are based on historical data and do not guarantee future performance
- Thorough validation is required before using for actual investment decisions
- Exchange rates are influenced by various economic and political factors
- Always seek professional financial advice

---

## 📄 License

**Custom License - Free for Personal Use, Commercial License Required**

This software is free to use for personal, educational, and non-commercial purposes. Commercial use requires a separate license agreement.

- ✅ **Free**: Personal use, education, research
- ❌ **Requires License**: Commercial use, production deployment, integration into commercial products

For commercial licensing inquiries, please contact: **hyun.lim@okkorea.net**

### Data Attribution

Exchange rate data collected from Investing.com and other public financial data sources. Accurate as of December 2024.

---

## 📞 Contact & Services

**Development Consulting & Outsourcing Available**

We provide professional consulting and development services for IoT, AI, and embedded systems projects.

### 👨‍💼 Project Manager Contact

- **Email**: hyun.lim@okkorea.net
- **Homepage**: https://www.okkorea.net
- **LinkedIn**: https://www.linkedin.com/in/aionlabs/

### 🛠️ Technical Expertise / 기술 전문 분야

- IoT System Design and Development / IoT 시스템 설계 및 개발
- Embedded Software Development / 임베디드 소프트웨어 개발 (Arduino, ESP32)
- AI Service Development / AI 서비스 개발 (LLM, MCP Agent)
- Cloud Service Architecture / 클라우드 서비스 구축 (Google Cloud Platform)
- Hardware Prototyping / 하드웨어 프로토타이핑

### 💼 Services / 서비스

**Technical Consulting / 기술 컨설팅**
- IoT project planning and design consultation / IoT 프로젝트 기획 및 설계 자문
- System architecture design / 시스템 아키텍처 설계

**Development Outsourcing / 개발 외주**
- Full-stack development from firmware to cloud / 펌웨어부터 클라우드까지 Full-stack 개발
- Proof of Concept (PoC) development / 개념 검증 개발
- Production-ready system development / 상용 시스템 개발

---

**Developed with ❤️ using Python, TensorFlow & LSTM**

*Last Updated: December 18, 2024*
