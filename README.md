# 📈 Stock Price Prediction with Time Series and Deep Learning

This project focuses on applying **time series analysis** and **deep learning** techniques to forecast the **monthly closing price** of a stock (e.g., AAPL). The primary goal is to predict the next month's closing price based on historical data from the past 10 years.

⚠️ **Note:** This project is currently under active development. New deep learning models, different dataset sizes, and feature engineering techniques are being explored and tested.

---

## 🎯 Goal

The main objective is to predict the **next month's closing price** of a given stock by leveraging historical data and applying various deep learning models.

---

## ⚙️ Pipeline Overview

1.  **Data Collection**:
    * Daily stock data for a specific ticker (e.g., "AAPL") is downloaded for the past 10 years using the `yfinance` library.
    * The raw daily data is saved to `data/dados_diarios.csv`.
2.  **Data Preprocessing**:
    * Daily data is aggregated to a monthly frequency, taking the last closing price of the month.
    * This monthly data is saved to `data/dados_mensais.csv`.
    * For LSTM models, data is checked for stationarity (using ADF and KPSS tests) and transformed (log, differencing) if necessary. Data is then scaled using MinMaxScaler.
3.  **Descriptive Analysis**:
    * Basic statistical analysis (mean, median, min, max, standard deviation, quantiles) is performed on monthly data using functions in `funcoes_auxiliares/descritive_data.py`.
4.  **Modeling**:
    * **LSTM (Long Short-Term Memory)**: An LSTM model is currently implemented to predict the next month's closing price. It uses a window of previous months' data (e.g., 6 months) as input. The predictions and evaluations are saved in `data/dados_mensais_predicted_lstm.csv` and `data/dados_mensais_evaluation_lstm.csv` respectively.
    * Placeholders for other models in `app.py` and `plots/plots.py` will be updated to reflect other deep learning approaches.
5.  **Evaluation**:
    * Models are evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Squared Error (MSE).
6.  **Visualization**:
    * Plots are generated to compare actual vs. predicted prices.

---

## 🛠️ Technologies Used

* **Python 3.8+**
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For preprocessing (MinMaxScaler) and model evaluation metrics.
* **Keras (TensorFlow backend)**: For building deep learning models (LSTM, etc.).
* **Matplotlib**: For creating visualizations.
* **yfinance**: To download historical stock market data.
* **Statsmodels**: For statistical tests like ADF and KPSS for stationarity.
* **Git & GitHub**: For version control and collaboration.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gcarucce10/stocks-analysis.git](https://github.com/gcarucce10/stocks-analysis.git)
    cd stocks-analysis
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    It's recommended to create a `requirements.txt` file. For now, you can install the necessary libraries based on the imports in the Python files (e.g., `pip install pandas numpy scikit-learn tensorflow matplotlib yfinance statsmodels`).
4.  **Run the main application:**
    ```bash
    python app.py
    ```
    This will:
    * Download the latest daily data for "AAPL" for the past 10 years.
    * Aggregate it into monthly data.
    * Train the LSTM model, predict the next month's price, and save the prediction.
    * Evaluate the LSTM model on historical data and save the results.

    *Note: The file paths in some scripts (e.g., `lstm/pipeline_lstm.py`, `plots/plots.py`) are currently absolute (`/home/useradd/...`). For better portability, these should be changed to relative paths.*

---

## 🔮 Next Steps & Future Development

This project is evolving. Key areas for future work include:

* **Implement Additional Deep Learning Models**:
    * Explore other recurrent neural networks (RNNs) like GRUs.
    * Investigate Convolutional Neural Networks (CNNs) for time series.
    * Test Attention-based models and Transformers if applicable.
* **Hyperparameter Tuning**: Optimize parameters for all deep learning models, including LSTM's window size, layers, activation functions, optimizers, etc.
* **Experimentation**:
    * Test with different lengths of historical data (e.g., 5, 15 years).
    * Explore different feature engineering techniques suitable for deep learning models.
    * Vary the stock ticker to test model generalization.
* **Refine Preprocessing**: Enhance stationarity checks and transformation strategies for deep learning inputs.
* **Code Improvement**:
    * Convert absolute file paths to relative paths.
    * Create a `requirements.txt` file for easier dependency management.
* **Deployment (Optional)**: Explore deploying a simple interface using Streamlit or a similar framework.

---
