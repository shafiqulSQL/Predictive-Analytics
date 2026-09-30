import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

# =====================================================================
# Sheet 1: Stock_Prices -- 3 tickers, ~750 trading days each, OHLC + volume
# =====================================================================
tickers = {
    "NOVATECH": {"start": 82.0, "drift": 0.00035, "vol": 0.018},   # tech growth stock
    "HARBORGO": {"start": 46.0, "drift": 0.00010, "vol": 0.012},   # steady logistics/retail
    "SOLARIX":  {"start": 121.0, "drift": -0.00015, "vol": 0.026},  # volatile clean-energy stock
}

n_days = 750
biz_days = pd.bdate_range("2022-01-03", periods=n_days)

stock_frames = []
for ticker, cfg in tickers.items():
    # geometric-brownian-motion-style path for the close price
    daily_returns = rng.normal(cfg["drift"], cfg["vol"], n_days)
    close = cfg["start"] * np.exp(np.cumsum(daily_returns))

    open_ = close * (1 + rng.normal(0, 0.004, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    volume = rng.integers(400_000, 1_100_000, n_days)

    df = pd.DataFrame({
        "date": biz_days,
        "ticker": ticker,
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": volume,
    })
    stock_frames.append(df)

stock_df = pd.concat(stock_frames, ignore_index=True)

# =====================================================================
# Sheet 2: Sales_Data -- 4 stores, 730 daily sales figures, trend + seasonality
# =====================================================================
stores = {
    "Uptown":   {"base": 640, "trend": 0.35, "weekly_amp": 90,  "yearly_amp": 60},
    "Lakeside": {"base": 505, "trend": 0.15, "weekly_amp": 70,  "yearly_amp": 45},
    "Harborview": {"base": 720, "trend": 0.50, "weekly_amp": 110, "yearly_amp": 80},
    "Prairie":  {"base": 430, "trend": -0.05, "weekly_amp": 55,  "yearly_amp": 35},
}

n_sales_days = 730
cal_days = pd.date_range("2023-01-01", periods=n_sales_days, freq="D")

sales_frames = []
for store, cfg in stores.items():
    t = np.arange(n_sales_days)
    trend = cfg["base"] + cfg["trend"] * t
    weekly = cfg["weekly_amp"] * np.sin(2 * np.pi * t / 7 + 1.2)
    yearly = cfg["yearly_amp"] * np.sin(2 * np.pi * t / 365.25)
    noise = rng.normal(0, 28, n_sales_days)
    sales = np.clip(trend + weekly + yearly + noise, 50, None)

    df = pd.DataFrame({"date": cal_days, "store": store, "sales": sales.round(1)})
    sales_frames.append(df)

sales_df = pd.concat(sales_frames, ignore_index=True)

# =====================================================================
# Sheet 3: Waiter_Tips -- 300 restaurant bills
# =====================================================================
n_tips = 300
sex = rng.choice(["Male", "Female"], n_tips, p=[0.56, 0.44])
smoker = rng.choice(["Yes", "No"], n_tips, p=[0.37, 0.63])
day = rng.choice(["Thur", "Fri", "Sat", "Sun"], n_tips, p=[0.24, 0.16, 0.35, 0.25])
time = np.where(np.isin(day, ["Sat", "Sun"]), rng.choice(["Lunch", "Dinner"], n_tips, p=[0.35, 0.65]),
                rng.choice(["Lunch", "Dinner"], n_tips, p=[0.55, 0.45]))
size = rng.choice([1, 2, 2, 2, 3, 4, 4, 5, 6], n_tips)

total_bill = rng.gamma(shape=6.0, scale=3.6, size=n_tips) + size * 2.3
total_bill = np.clip(total_bill, 5, 60).round(2)

# tip rate depends mildly on size, time, and smoker status, plus noise
base_rate = 0.16
rate = (base_rate
        + np.where(time == "Dinner", 0.01, -0.005)
        + np.where(smoker == "Yes", -0.01, 0.0)
        - 0.004 * (size - 2).clip(min=0)
        + rng.normal(0, 0.03, n_tips))
tip = np.clip(total_bill * rate, 0.5, None).round(2)

tips_df = pd.DataFrame({
    "total_bill": total_bill, "tip": tip, "sex": sex, "smoker": smoker,
    "day": day, "time": time, "size": size,
})

# =====================================================================
# Write all three sheets to one workbook
# =====================================================================
out_path = "predictive_analytics_dataset_v2.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    stock_df.to_excel(writer, sheet_name="Stock_Prices", index=False)
    sales_df.to_excel(writer, sheet_name="Sales_Data", index=False)
    tips_df.to_excel(writer, sheet_name="Waiter_Tips", index=False)

print("Stock_Prices:", stock_df.shape, stock_df["ticker"].unique())
print("Sales_Data:", sales_df.shape, sales_df["store"].unique())
print("Waiter_Tips:", tips_df.shape)
print("written to", out_path)
