# FX-Forecasting-with-LSTM

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-green.svg)](https://gradio.app/)
[![Pandas](https://img.shields.io/badge/Pandas-1.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.0+-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-blue.svg)](https://matplotlib.org/)

## Project Description

This project develops a machine learning model to predict USD/KRW exchange rates using Long Short-Term Memory (LSTM) neural networks. By analyzing various economic indicators such as the US Dollar Index (DXY) and Commodity Research Bureau (CRB) index, the model forecasts future exchange rate movements.

The project includes:
- **Data Collection**: Automated scraping of economic indicators from financial websites
- **Data Preprocessing**: Time series alignment, missing value handling, and feature scaling
- **Model Training**: LSTM-based forecasting with hyperparameter tuning
- **Web Interface**: Interactive Gradio app for model training and prediction visualization
- **Evaluation**: Performance metrics including MAE and R² score

## Features

- Real-time economic data collection from multiple sources
- LSTM model training with customizable parameters
- Interactive web interface for easy model deployment
- Comprehensive data preprocessing pipeline
- Visualization of prediction results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ganjjiang/FX-Forecasting-with-LSTM.git
cd FX-Forecasting-with-LSTM
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
```bash
python src/fx_forecasting/data_collection.py
```

### Run the Gradio Web App
```bash
python src/fx_forecasting/gradio_app.py
```

### Run Tests
```bash
pytest tests/
```

## Project Structure

```
├── src/
│   └── fx_forecasting/
│       ├── __init__.py
│       ├── data_collection.py    # Data scraping and collection
│       ├── preprocessing.py      # Data cleaning and preprocessing
│       ├── modeling.py          # LSTM model implementation
│       └── gradio_app.py        # Web interface
├── tests/                       # Unit tests
├── screenshots/                 # Screenshots of the application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras**: For LSTM model implementation
- **Gradio**: Web interface for model interaction
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Data preprocessing
- **Requests & BeautifulSoup**: Web scraping

## Background

Exchange rates play a crucial role in international trade and financial transactions. However, exchange rates are difficult to predict and tend to be volatile, making prediction increasingly necessary. There are various methods for prediction, but machine learning-based approaches have gained attention recently. Therefore, this project aims to develop an exchange rate prediction model using machine learning algorithms.

## Project Objectives
- **Exchange Rate Prediction Model Development**: Develop and evaluate an exchange rate prediction model using machine learning algorithms.
- **Identification of Predictive Variables**: Identify which variables have the most significant impact on exchange rate prediction, assuming various variables influence it.
- **Model Interpretation**: Analyze the developed model to understand how each variable affects exchange rate prediction.

## Analysis Procedure

1. **Exchange Rate Characteristics Analysis**: Identify changes in factors affecting exchange rate fluctuations and analyze the impact of these changes.
2. **Economic Indicators Analysis**: Analyze economic indicators related to exchange rates (such as real GDP, inflation, imports/exports) to understand their relationship with exchange rates.
3. **Data Collection**: Visualize collected data to discover various insights and utilize them for reports and presentations.
4. **Data Preprocessing**: Collect, clean, and extract necessary data for optimal preprocessing.
5. **Data Analysis**: Analyze correlations between collected data to select variables for the final model.
6. **Model Creation**: Fine-tune parameters with selected variables to find the minimum error value.

## Screenshots

![Main Interface](screenshots/main.png)

## Key Analysis Features
### Data Collection
- Web scraping for exchange rate information using Requests, Selenium, and Beautiful Soup
### Data Preprocessing
- Time stamp alignment using Pandas library
- Other data preprocessing using scikit-learn
### Data Analysis
- Pattern recognition in time series data using combined models such as ARIMA and Linear Regression
- Model performance improvement through hyperparameter tuning and cross-validation
- Visualization of model prediction results using matplotlib, seaborn, etc.

---

## EDA and Overview LSTM

# Summary of EDA and Exchange Rate Prediction Model (Based on PPT)

---

## 1. Exploratory Data Analysis (EDA)

### 1.1 Purpose
EDA was conducted to analyze the USD/KRW exchange rate as a **multivariate macroeconomic system**, rather than a series driven by technical trading indicators.  
The goal was to identify patterns, dependencies, and key influencing factors to support subsequent predictive modeling.

---

### 1.2 Data Collection and Initial Exploration

- **Time Coverage**
  - Data collected from 1998 onward
  - Mixed frequencies: daily, monthly, quarterly

- **Analytical Scope**
  - Technical indicators (MA, RSI, etc.) explicitly excluded
  - Focus on macroeconomic, financial, international, and sentiment variables

- **Variable Categories**
  - **Economic Factors**
    - GDP, economic growth rate, government expenditure, gross national income
    - Money supply (M1, M2, M3)
  - **Financial Factors**
    - Policy interest rate, bond yields
    - Foreign exchange reserves
    - Foreign ownership ratio
    - Equity indices (KOSPI, S&P 500)
  - **International Factors**
    - CRB Index
    - Commodity prices (gold, crude oil, natural gas)
    - Global semiconductor index
  - **Political & Sentiment Factors**
    - VIX
    - US Dollar Index (DXY)
    - News sentiment index
    - Consumer sentiment index
    - Governing party indicator

- **Objective**
  - To explore multivariate relationships driving exchange rate movements

---

### 1.3 Data Preprocessing

- **Temporal Alignment**
  - All variables aligned to monthly frequency
  - Daily data aggregated by monthly mean
  - Quarterly data interpolated to monthly values

- **Date Standardization**
  - Unified to `YYYY-MM` format

- **Missing Value Handling**
  - Forward fill and linear interpolation applied

- **Scaling**
  - Standard Scaling applied to all variables

- **Final Dataset**
  - Shape: **(303, 84)**
    - 303 monthly observations
    - 84 independent variables
  - Problem formulation: multivariate regression

---

### 1.4 Variable Relationship Analysis

- **Models Applied**
  - Linear Regression
  - Ridge
  - Lasso
  - Elastic Net
  - Random Forest

- **Evaluation Metric**
  - R² score

- **Findings**
  - High explanatory power observed
  - Significant overfitting risk identified

- **Conclusion**
  - USD/KRW exchange rate behaves as a complex system
  - Single-variable explanations are insufficient

---

## 2. Exchange Rate Prediction Model

### 2.1 Problem Definition and Approach

- **Target Variable**
  - USD/KRW exchange rate

- **Objective**
  - Forecast exchange rates using macroeconomic, financial, international, and sentiment indicators

- **Approach**
  - Multivariate time-series modeling
  - Baseline machine learning models followed by deep learning (LSTM)

---

### 2.2 Modeling Workflow

- **Baseline Model Comparison**
  - Linear Regression, Ridge, Lasso, Elastic Net, Random Forest
  - R²-based evaluation
  - Overfitting identified as a limitation

- **Cross-Validation Strategy**
  - Conventional CV rejected due to time-series leakage
  - Time-series cross-validation applied
  - Sequential Train → Validation split

---

## 3. LSTM Model Implementation

### 3.1 Rationale for LSTM

- Designed to capture long-term dependencies in time-series data
- Supported by prior research demonstrating superior performance
- Role defined as a **multivariate time-series pattern learner**
- Focus on short-term prediction (1–2 days ahead)

---

### 3.2 Input Variable Selection

- **Initial Candidates**
  - USD/KRW
  - US Dollar Index (DXY)
  - VIX
  - CRB Commodity Index

- **Final Variables**
  - USD/KRW
  - DXY
  - CRB Index
  - (VIX excluded due to MAE increase)

- **Dataset Characteristics**
  - Daily data since 1998
  - Early 1998 outliers removed
  - Monthly data replaced with daily data to increase sample size

- **Input Shape**
  - `(samples, 10, 3)`
    - Past 10 days
    - 3 features (USD/KRW, CRB, DXY)

---

### 3.3 Data Preprocessing for LSTM

- **Missing Values**
  - Rows with all-null values removed
  - Forward fill selected over interpolation due to better performance

- **Scaling**
  - Manual standardization:
    - `(value - mean) / std`
  - Inverse transformation:
    - `prediction * std + mean` (USD/KRW statistics)

- **Train / Validation Split**
  - 90% training, 10% validation
  - Sliding window generation:
    - History size: 10
    - Target size: 1
    - Step size: 1

### 3.4 Model Architecture

- **Framework**
  - TensorFlow / Keras

- **Network Structure**
  ```python
  Sequential(
      LSTM(200, activation="tanh", input_shape=(10, 3)),
      Dense(1)
  )
  ```
  - Configuration
    - Optimizer: RMSprop
    - Loss Function: MAE
### 3.5 Training Strategy

- Batch size: 32  
- Epochs: 100  
- EarlyStopping applied to monitor validation loss and prevent overfitting  
- ModelCheckpoint used to save the best-performing model weights based on minimum validation loss  

---

### 3.6 Model Evaluation

- **Evaluation Metrics**
  - MAE (Mean Absolute Error): approximately 3.8–3.9 KRW
  - R² score for overall explanatory power
  - Error distribution:
    - Maximum error: ~34 KRW
    - Majority of errors: below 23 KRW

- **Visualization**
  - Training vs. validation loss curves
  - Absolute prediction error scatter plots
  - Actual vs. predicted USD/KRW exchange rate time series

---

### 3.7 Prediction Process

- **Next-Day Forecasting**
  - Standardize the most recent 10 days of input data
  - Reshape input to `(1, 10, 3)`
  - Generate prediction using the trained LSTM model
  - Apply inverse scaling to recover the predicted exchange rate level
  - Output: next-day USD/KRW exchange rate estimate

- **Trend Estimation**
  - Apply a 2-day moving average to recent data
  - Infer trend direction from the sign of change between predicted and previous values

---

## 4. Performance and Limitations

- **Achievements**
  - High short-term prediction accuracy with MAE around 3.8 KRW
  - Enhanced directional interpretation through moving average integration

- **Limitations**
  - Prediction error increases with longer forecast horizons
  - Limited variable set restricts broader macroeconomic interpretation

- **Future Work**
  - Ensemble modeling with alternative architectures
  - Incorporation of multiple moving average horizons (2, 5, 10 days)
  - Deployment as a web-based exchange rate forecasting service





## Sources
Korea Capital Market Institute https://www.kcif.or.kr/front/board/boardList.do?intSection1=2&intSection2=4&intBoardID=1 <br>
Bank of Korea Economic Statistics System http://ecos.bok.or.kr <br>
IMF http://www.imf.org/en/data <br>
World Bank https://data.worldbank.org <br>
BIS https://www.bis.org/ <br>
Economic Policy Uncertainty http://www.policyuncertainty.com/ <br>
Bloomberg, CEIC <br>
Ministry of Trade, Industry and Energy https://www.motie.go.kr/motie/py/sa/investstatse/investstats.jsp <br>
Real-time exchange rate data, Investing.com <br>
Financial data Yahoo Finance <br>
Quandl for various fields including finance, economy, and politics <br>
Federal Reserve Economic Data FRED <br>
Monthly exchange rate information Korea Financial Investment Association (KOFIA) <br>

---


## 📞 문의하기

**개발 관련 컨설팅 및 외주 받습니다.**

### 👨‍💼 프로젝트 관리자 연락처

**Email**: [hyun.lim@okkorea.net](mailto:hyun.lim@okkorea.net)  
**Homepage**: [https://www.okkorea.net](https://www.okkorea.net)  
**LinkedIn**: [https://www.linkedin.com/in/aionlabs/](https://www.linkedin.com/in/aionlabs/)  

### 🛠️ 전문 분야

- **IoT 시스템 설계 및 개발**
- **임베디드 소프트웨어 개발** (Arduino, ESP32)
- **AI 서비스 개발** (LLM, MCP Agent)
- **클라우드 서비스 구축** (Google Cloud Platform)
- **하드웨어 프로토타이핑**

### 💼 서비스

- **기술 컨설팅**: IoT 프로젝트 기획 및 설계 자문
- **개발 외주**: 펌웨어부터 클라우드까지 Full-stack 개발
- **교육 서비스**: 임베디드/IoT 개발 교육 및 멘토링

