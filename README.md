# Binance API Utilities for Trading Automation

## Overview

This repository contains a collection of **Python utilities** for interacting with the **Binance API**, focusing on **trade execution, risk management, and market data retrieval**. The provided functions demonstrate best practices in **API integration, data processing, order execution, and risk calculation**.

> 🚀 **This is not a full trading bot** but a showcase of my expertise in **Python programming, algorithmic trading, and financial data processing**.

## **Key Features**
- 📊 **Market Data Retrieval** – Fetches real-time and historical market data, including **candlesticks (klines), open interest, and order book snapshots**.
- ⚡ **Trade Execution & Order Management** – Implements **automated order placement, status updates, and cancellations** with **robust error handling**.
- 🔍 **Risk Management & Trade Calculations** – Computes **stop-loss, take-profit levels, trade commissions, and position sizing** based on ATR and other risk parameters.
- 📂 **Data Storage & Processing** – Saves historical data in **CSV format** for backtesting and logs order execution details.
- 🛠️ **Optimized for Performance & Reliability** – Includes **retry mechanisms, logging, and data validation** to ensure stable API interactions.

## **File Structure**
This repository consists of three core Python modules:

### **1️⃣ Binance Trade Manager**
📌 **File:** `binance_trade_manager.py`  
🔹 Functions for managing trade execution, order handling, and real-time updates.

| Function | Description |
|----------|------------|
| `update_orders_info(client, trade)` | Fetches and updates trade order statuses (limit, stop-loss, take-profit). |
| `open_orders(client, trade)` | Places **limit, stop-loss, and take-profit orders** with automatic retries. |
| `cancel_orders(client, trade)` | Cancels all pending orders for a given trade. |
| `close_orders(client, trade)` | Closes an open trade by executing a **market order** if necessary. |

🔹 **Key Features**:
- **Robust error handling & retries** for API requests.
- **Automated logging** using Python’s built-in logging module.
- **Optimized trade closing strategy** with dynamic market orders.

---

### **2️⃣ Risk and Trade Parameters**
📌 **File:** `risk_and_trade_params.py`  
🔹 Functions for **risk assessment, trade sizing, and profit/loss calculations**.

| Function | Description |
|----------|------------|
| `calc_trade_res(entry_price, exit_price, amount, direction)` | Computes **trade results** based on entry/exit prices. |
| `stop_loss(atr, atr_multiplier, min_stop_loss, max_stop_loss=None)` | Calculates **stop-loss levels** using **ATR (Average True Range)**. |
| `take_profit_price(entry_price, stop_loss, risk_profit_coef)` | Computes **take-profit price** based on risk/reward ratio. |
| `calc_commission(entry_price, exit_price, amount_per_trade, taker_rate)` | Calculates **trading commission costs**. |
| `create_new_trade(symbol, timestamp, direction, entry_price, atr, usd_per_trade)` | Creates a structured trade dictionary with predefined risk parameters. |

🔹 **Key Features**:
- Uses **ATR-based stop-loss calculations** for adaptive risk management.
- Computes **trading fees and position sizing** based on user-defined risk constraints.
- Dynamically adjusts **trade entry and exit points** based on risk-reward ratios.

---

### **3️⃣ Binance Market Data Retrieval**
📌 **File:** `binance_get_data.py`  
🔹 Functions for fetching, validating, and processing **market data from Binance API**.

| Function | Description |
|----------|------------|
| `get_data_last_raw(client, symbol, interval_str, start_time, limit=2)` | Fetches the **latest spot & futures candlestick data**. |
| `get_klines_data(client, symbol, interval_str, start_time_str, limit=1000)` | Retrieves historical **klines (candlestick) data**. |
| `get_sumOpenInterest(client, symbol, interval_str, start_time_str=None)` | Fetches **futures open interest** data and processes missing values. |
| `save_orderbook_to_csv(client, symbol, time_UTC, path, limit_spot=5000, limit_perp=1000)` | Saves **order book snapshots** to CSV for further analysis. |

🔹 **Key Features**:
- Implements **data validation and error handling** to ensure **data integrity**.
- Uses **time synchronization and re-indexing** to align datasets.
- Logs **missing data points** and applies **interpolation techniques** for data smoothing.
- Saves **order book and historical trade data** to CSV for backtesting and analysis.

## **Why This Project?**
This repository highlights my skills as a **Software Engineer** specializing in **financial trading automation**. Specifically, it showcases:

✅ **Advanced Python development** – Efficient, structured, and reusable code.  
✅ **API interaction & data processing** – Handling **REST API requests, data validation, and error management**.  
✅ **Trading risk management** – Implementation of **ATR-based stop-loss, take-profit, and trade sizing strategies**.  
✅ **System reliability** – **Retry mechanisms, logging, and data integrity checks**.  
✅ **Automated data storage** – CSV export and structured **historical market data handling**.

## **Installation & Usage**
### **1️⃣ Install Dependencies**
```bash
pip install pandas python-binance numpy tabulate
```
All dependencies are listed in `requirements.txt`.  

## **Example Usage**

```bash
from binance.client import Client
from binance_get_data import get_klines_data

client = Client(api_key="your_api_key", api_secret="your_api_secret")
df = get_klines_data(client, symbol="BTCUSDT", interval_str="1h", start_time_str="01.01.2024")
print(df.head())
```

## **Logging & Data Storage**

- **Logs all API interactions**, including errors, retries, and data validation issues.
- **Stores order book and market data in CSV files** for historical analysis and backtesting.
- Uses **timestamp-based filenames** to prevent overwriting and maintain data versioning.

## ⚠️ Disclaimer  
This code is provided for **educational purposes only** and is not intended for live trading. Always use caution and perform extensive testing before executing real trades.

## 📌 Technologies Used
- Programming Language: Python 🐍
- Data Handling & Processing: Pandas, NumPy 📊
- API Interaction: Binance API 🔗
- Logging & Debugging: Python logging module for robust monitoring 📜
- Error Handling & Reliability: Retry mechanisms, exception handling ⚠️
- Data Storage: CSV file management for historical market data 📂
- Risk Management & Trading Models: ATR-based stop-loss, take-profit strategies 📈

## Author

- **Ilgiz Almukhametov**  
- **GitHub:** https://github.com/ilgiz-almv
- **LinkedIn:** https://www.linkedin.com/in/ilgiz-almv/

## License
This project is licensed under the **MIT License**.

**GitHub Repository:** https://github.com/ilgiz-almv/binance_utils
