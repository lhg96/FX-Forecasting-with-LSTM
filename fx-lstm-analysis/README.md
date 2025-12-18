# FX LSTM Analysis - Exchange Rate Forecasting

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10+-blue.svg)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Custom-red.svg)](#-license)

A modular USD/KRW exchange rate forecasting system using LSTM (Long Short-Term Memory) neural networks.

## 📋 Overview

This project implements a complete pipeline for USD/KRW exchange rate forecasting using LSTM neural networks. The system collects real-time forex data, preprocesses it, trains an LSTM model, and generates comprehensive analytical visualizations.

### Key Features

- **🌐 Automated Data Collection**: Web scraping from Investing.com for USD/KRW, Dollar Index (DXY), and CRB Index
- **🔄 Data Preprocessing**: Standard normalization and train/validation splitting (90/10)
- **🧠 LSTM Architecture**: 200-unit LSTM layer with early stopping and model checkpointing
- **📊 Comprehensive Evaluation**: MAE, MSE, RMSE, R² Score, trend analysis
- **📈 Rich Visualizations**: Training history, predictions, error analysis, moving averages, correlation heatmaps
- **🎯 Next-Day Prediction**: Automated forecasting with trend direction analysis

## 🏗️ Project Structure

```
fx-lstm-analysis/
├── src/
│   ├── data_collector.py        # Web scraping module
│   ├── data_preprocessor.py     # Data normalization & splitting
│   ├── lstm_model.py            # LSTM model definition
│   ├── model_evaluator.py       # Performance evaluation
│   ├── visualizer.py            # Visualization functions
│   ├── main.py                  # Pipeline orchestration
│   └── generate_sample_data.py  # Synthetic data generation
├── data/                         # Data storage
├── models/                       # Trained model checkpoints
├── output/                       # Generated visualizations
├── screenshots/                  # Result screenshots
├── requirements.txt              # Python dependencies
├── setup.sh                      # Automated setup script
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd fx-lstm-analysis

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Generate sample data (recommended)
cd src
python generate_sample_data.py

# Run complete pipeline
python main.py
```

## 📦 Dependencies

- **Python**: 3.12+
- **TensorFlow**: 2.20.0 (with Keras 3.13.0)
- **Pandas**: 2.3.3
- **NumPy**: 2.3.5
- **Matplotlib**: 3.10.8
- **Scikit-learn**: 1.8.0
- **BeautifulSoup4**: For web scraping
- **openpyxl**: Excel file handling

## 🔧 Module Details

### 1. Data Collector (`data_collector.py`)
- Scrapes USD/KRW exchange rates from Investing.com
- Collects Dollar Index (DXY) and CRB Commodity Index
- Merges datasets by date with automatic column handling
- Error handling for website structure changes

### 2. Data Preprocessor (`data_preprocessor.py`)
- Standard normalization (z-score)
- Train/validation split (90/10)
- Denormalization for final predictions
- Stores mean/std for inverse transformation

### 3. LSTM Model (`lstm_model.py`)
- **Architecture**: Sequential([LSTM(200, tanh), Dense(1)])
- **Optimizer**: RMSprop
- **Loss Function**: Mean Absolute Error (MAE)
- **Callbacks**: EarlyStopping(patience=40), ModelCheckpoint
- Configurable hyperparameters via `ModelConfig` class

### 4. Model Evaluator (`model_evaluator.py`)
- **Metrics**: MAE, MSE, RMSE, R² Score
- **Trend Analysis**: Moving averages (20-day, 50-day, 100-day)
- **Next-Day Prediction**: Automated forecasting with confidence estimation
- Directional accuracy measurement

### 5. Visualizer (`visualizer.py`)
- Training history plots (loss curves)
- Actual vs predicted comparisons
- Prediction error distribution
- Moving average trend analysis
- Correlation heatmaps
- Korean font support (AppleGothic for macOS)

### 6. Main Pipeline (`main.py`)
- Orchestrates 7-step workflow
- Interactive and automated modes
- Error handling and logging
- Auto-mode for CI/CD integration

## 📊 Performance Metrics

Based on 6978 samples (1998-2024):

- **Training/Validation**: 90/10 split
- **Training Epochs**: 57/100 (early stopping)
- **MAE**: 8.75 KRW
- **R² Score**: 0.99251
- **Accuracy**: 99.85% within 50 KRW threshold
- **Next-Day Trend**: Directional prediction with moving average confirmation

## 📈 Output Examples

The system generates 6 key visualizations:

1. **Training History**: Loss curves over epochs
2. **Predictions**: Actual vs predicted overlay
3. **Error Analysis**: Residual distribution
4. **Moving Averages**: 20/50/100-day trend lines
5. **Correlation Heatmap**: Feature relationships
6. **Scatter Plot**: Prediction accuracy visualization

See [`screenshots/`](screenshots/) folder for sample outputs.

## 🛠️ Troubleshooting

### Common Issues

1. **Web Scraping Failures**
   - Solution: Use `generate_sample_data.py` to create synthetic data
   - Investing.com structure may change, causing scraper to return incomplete data

2. **Package Compatibility** (Python 3.12)
   - Ensure numpy>=1.26.0, tensorflow>=2.15.0
   - AttributeError with pkgutil: Upgrade numpy to 1.26+

3. **Model Checkpoint Errors**
   - Use `.weights.h5` extension for TensorFlow 2.13+
   - Path: `../models/best_model.weights.h5`

4. **Interactive Mode Errors**
   - EOFError in non-interactive environments
   - main.py includes `sys.stdin.isatty()` checks for automation

## 🔄 Automated Setup

```bash
# Use setup script for one-command installation
bash setup.sh
```

## 📝 License

**Custom License - Free for Personal Use, Commercial License Required**

This software is available for personal, educational, and research purposes at no cost.
For commercial use, enterprise deployment, or integration into commercial products, please contact for licensing.

© 2024 Hyun Lim. All rights reserved.

## 📧 Contact & Services

### Author
**Hyun Lim**  
📧 Email: hyun.lim@okkorea.net

### Technical Expertise
- **AI/ML Development**: TensorFlow, PyTorch, LSTM, Transformer architectures
- **Financial Modeling**: Time series forecasting, quantitative analysis
- **Cloud Architecture**: AWS, Azure, Kubernetes deployment
- **Full-Stack Development**: Python, Node.js, React, microservices

### Services Available
- Custom AI model development and fine-tuning
- Financial forecasting system implementation
- Cloud infrastructure setup and optimization
- Technical consulting for AI/ML projects
- Enterprise software architecture design

---

### 전문 분야
- **AI/ML 개발**: TensorFlow, PyTorch, LSTM, Transformer 아키텍처
- **금융 모델링**: 시계열 예측, 계량 분석
- **클라우드 아키텍처**: AWS, Azure, Kubernetes 배포
- **풀스택 개발**: Python, Node.js, React, 마이크로서비스

### 제공 서비스
- 맞춤형 AI 모델 개발 및 파인튜닝
- 금융 예측 시스템 구현
- 클라우드 인프라 설정 및 최적화
- AI/ML 프로젝트 기술 컨설팅
- 엔터프라이즈 소프트웨어 아키텍처 설계

---

**For inquiries regarding commercial licensing, custom development, or consulting services:**  
�� hyun.lim@okkorea.net
