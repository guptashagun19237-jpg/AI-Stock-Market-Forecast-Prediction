import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------- Page Settings ---------------- #
st.set_page_config(
    page_title="AI Stock Market Forecast",
    page_icon="📈",
    layout="wide"
)

# ---------------- Sidebar ---------------- #
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Stock Analysis",
        "Prediction",
        "About Project"
    ]
)
st.sidebar.header("Stock Settings")
# ---------------- Main Title ---------------- #
st.title("📈 AI Stock Market Forecast Prediction")

st.write(
    "This project predicts stock prices using Python, Machine Learning, and Deep Learning."
)

stock_symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="AAPL"
)
start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2020,1,1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)
stock = yf.download(
    stock_symbol,
    start=start_date,
    end=end_date,
) 
try:
    ticker = yf.Ticker(stock_symbol)
    info = ticker.info
except:
    info={}
if hasattr(stock.columns, "nlevels") and stock.columns.nlevels > 1:
    stock.columns = stock.columns.get_level_values(0)

if stock.empty:
    st.error("No data found")
    st.stop()
current_price = float(stock["Close"].iloc[-1])

high_price = float(stock["High"].max())

low_price = float(stock["Low"].min())

volume = int(stock["Volume"].iloc[-1])


stock["MA50"] = stock["Close"].rolling(50).mean()

stock["MA200"] = stock["Close"].rolling(200).mean()

st.sidebar.markdown("---")

st.sidebar.info(
    """
    📈 AI Stock Market Forecast

    Version: 1.0

    Developer: Shagun Gupta

    Technology:
    - Python
    - Streamlit
    - Plotly
    - Scikit-learn
    - Yahoo Finance
    """
)

if page == "Dashboard":

    st.header("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("Highest Price", f"${high_price:.2f}")
    col3.metric("Lowest Price", f"${low_price:.2f}")
    col4.metric("Volume", f"{volume:,}") 

    st.subheader("🏢 Company Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Company Name:**", info.get("longName", "N/A"))
        st.write("**Sector:**", info.get("sector", "N/A"))
        st.write("**Industry:**", info.get("industry", "N/A"))
        st.write("**Country:**", info.get("country", "N/A"))

    with col2:
        st.write("**Market Cap:**", info.get("marketCap", "N/A"))
        st.write("**P/E Ratio:**", info.get("trailingPE", "N/A"))
        st.write("**52 Week High:**", info.get("fiftyTwoWeekHigh", "N/A"))
        st.write("**52 Week Low:**", info.get("fiftyTwoWeekLow", "N/A"))
        st.subheader("📈 Stock Closing Price")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock.index,
            y=stock["Close"],
            mode="lines",
            name="Closing Price",
            line=dict(color="blue", width=3)
    )
)
    fig.add_trace(
        go.Scatter(
            x=stock.index,
            y=stock["MA50"],
            mode="lines",
            name="50 Day MA",
            line=dict(color="orange", width=2)
    )
)

    fig.add_trace(
        go.Scatter(
            x=stock.index,
            y=stock["MA200"],
            mode="lines",
            name="200 Day MA",
            line=dict(color="green", width=2)
    )
)
    fig.update_layout(
        title=f"{stock_symbol} Closing Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=600,
        hovermode="x unified"
     )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Stock Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
    "Average Closing Price",
    f"${stock['Close'].mean():.2f}"
)

    col2.metric(
    "Maximum Closing Price",
    f"${stock['Close'].max():.2f}"
)

    col3.metric(
    "Minimum Closing Price",
    f"${stock['Close'].min():.2f}"
)
    st.subheader("📈 Stock Performance Summary")

    total_return = (
    (stock["Close"].iloc[-1] - stock["Close"].iloc[0])
    / stock["Close"].iloc[0]
)     * 100

    if total_return > 0:
        st.success(f"Overall Return: {total_return:.2f}% 📈")
    else:
        st.error(f"Overall Return: {total_return:.2f}% 📉")
    st.subheader("🏆 Best & Worst Trading Day")

    highest_day = stock["High"].idxmax()
    lowest_day = stock["Low"].idxmin()

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Highest Price Date: {highest_day.date()}")

    with col2:
        st.error(f"Lowest Price Date: {lowest_day.date()}")

    st.subheader("🕯️ Candlestick Chart")

    candle = go.Figure()

    candle.add_trace(
        go.Candlestick(
            x=stock.index,
            open=stock["Open"],
            high=stock["High"],
            low=stock["Low"],
            close=stock["Close"],
            name="Candlestick"
    )
)

    candle.update_layout(
         title=f"{stock_symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
)

    st.plotly_chart(
        candle,
        use_container_width=True,
        key="candlestick_chart"
)

    st.subheader("📊 Trading Volume")

    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=stock.index,
            y=stock["Volume"],
            name="Volume"
    )
)

    volume_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Volume"
)

    st.plotly_chart(volume_fig, use_container_width=True)

    # ---------------- Historical Data ---------------- #

    st.subheader("📋 Historical Stock Data")

    display_data = stock.copy()

    display_data = display_data.reset_index()

    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
)
 
    st.subheader("📥 Download Stock Data")

    csv = stock.to_csv(index=False).encode("utf-8")

    st.download_button(
    label="📄 Download CSV",
    data=csv,
    file_name=f"{stock_symbol}_historical_data.csv",
    mime="text/csv",
)
 
    st.subheader("💡 AI Stock Recommendation")

    latest_close = stock["Close"].iloc[-1]
    ma50 = stock["MA50"].iloc[-1]
    ma200 = stock["MA200"].iloc[-1]

    if latest_close > ma50 and ma50 > ma200:
        st.success("🟢 BUY Recommendation")
        st.write("The stock is showing a strong upward trend.")

    elif latest_close < ma50 and ma50 < ma200:
        st.error("🔴 SELL Recommendation")
        st.write("The stock is showing a downward trend.")

    else:
         st.warning("🟡 HOLD Recommendation")
         st.write("The stock is moving sideways. Wait before buying or selling.")

elif page == "Stock Analysis":

    st.header("📈 Stock Analysis")
    st.subheader("📋 Historical Stock Data")

    st.dataframe(stock)
    st.subheader("📌 Last 10 Records")

    st.dataframe(stock.tail(10))
    st.write("Stock charts will appear here.")

    # ---------------- Download Historical Data ---------------- #

    st.subheader("⬇️ Download Stock Data")

    csv = stock.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Historical Data (CSV)",
        data=csv,
        file_name=f"{stock_symbol}_historical_data.csv",
        mime="text/csv"
)

elif page == "Prediction":

    st.header("🤖 AI Stock Price Prediction")

    # Copy stock data
    prediction_data = stock.copy()

    # Target column (Next Day Closing Price)
    prediction_data["Prediction"] = prediction_data["Close"].shift(-1)

    # Last row remove because prediction is empty
    prediction_data.dropna(inplace=True)

    # Features (Today's Close Price)
    X = prediction_data[["Close"]]

    # Target (Tomorrow's Close Price)
    y = prediction_data["Prediction"]

    st.write("### Dataset Preview")
    st.dataframe(prediction_data.head())
    
    # ---------------- Train Test Split ---------------- #

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
)

# ---------------- Linear Regression Model ---------------- #

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    st.success("✅ Machine Learning Model Trained Successfully!")

    st.subheader("📈 Prediction Sample")

    prediction_df = prediction_data.loc[X_test.index].copy()

    prediction_df["Predicted Price"] = predictions

    st.dataframe(
        prediction_df[
            ["Close", "Prediction", "Predicted Price"]
        ].head(10)
)
 
    st.subheader("📊 Actual vs Predicted Stock Price")

    prediction_graph = go.Figure()

    prediction_graph.add_trace(
        go.Scatter(
            x=list(range(len(y_test))),
            y=y_test.values,
            mode="lines",
            name="Actual Price",
            line=dict(color="blue", width=3)
    )
)

    prediction_graph.add_trace(
        go.Scatter(
            x=list(range(len(predictions))),
            y=predictions,
            mode="lines",
            name="Predicted Price",
            line=dict(color="red", width=3)
    )
)

    prediction_graph.update_layout(
        title="Actual vs Predicted Stock Price",
        xaxis_title="Data Points",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=600,
        hovermode="x unified"
)

    st.plotly_chart(
    prediction_graph,
    use_container_width=True,
    key="prediction_graph"
)

# ---------------- Model Performance ---------------- #

    st.subheader("📊 Model Performance")

# Calculate Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

# Display Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
    "MAE",
    f"{mae:.2f}"
)

    col2.metric(
    "RMSE",
    f"{rmse:.2f}"
)

    col3.metric(
    "R² Score",
    f"{r2:.4f}"
)

# ---------------- Predict Tomorrow's Price ---------------- #

    st.subheader("🔮 Tomorrow's Stock Price Prediction")

    if st.button("Predict Next Day Price"):

        latest_close = np.array([[stock["Close"].iloc[-1]]])

        tomorrow_price = model.predict(latest_close)[0]

        st.success(
         f"📈 Predicted Closing Price for Next Trading Day: ${tomorrow_price:.2f}"
)

st.subheader("💰 Investment Calculator")

investment = st.number_input(
    "Enter Investment Amount ($)",
    min_value=100.0,
    value=1000.0,
    step=100.0
)

if st.button("Calculate Investment"):

    current_price = float(stock["Close"].iloc[-1])

    shares = investment / current_price

    st.success(f"You can buy approximately {shares:.2f} shares.")

    st.info(f"Current Stock Price: ${current_price:.2f}")

    st.write(f"Investment Value: ${investment:.2f}")

elif page == "About Project":

    st.header("ℹ️ About Project")

    st.write("""
    ## 📌 Project Overview

    AI Stock Market Forecast Prediction is a web application developed using
    Python and Streamlit. It helps users analyze stock market data,
    visualize trends, and predict the next trading day's closing price
    using a Machine Learning model.

    The application fetches real-time historical stock data from Yahoo Finance
    and provides interactive charts for better analysis.
    """)

    st.subheader("🚀 Features")

    st.markdown("""
    - 📊 Interactive Dashboard
    - 📈 Closing Price Chart
    - 📉 50-Day & 200-Day Moving Average
    - 🕯️ Candlestick Chart
    - 📦 Trading Volume Analysis
    - 🏢 Company Information
    - 📋 Historical Stock Data
    - 📥 Download Data as CSV
    - 🤖 Machine Learning Stock Price Prediction
    - 📊 Actual vs Predicted Comparison
    - 📏 Model Performance Metrics
    """)

    st.subheader("🛠 Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Yahoo Finance (yfinance)
    - Scikit-learn
    """)

    st.subheader("🤖 Machine Learning Model")

    st.info("""
    This project uses the Linear Regression algorithm from Scikit-learn
    to predict the next trading day's closing stock price based on
    historical stock data.
    """)

    st.subheader("📈 Future Scope")

    st.markdown("""
    - LSTM Deep Learning Prediction
    - Live Stock Market Data
    - Buy/Sell Recommendation
    - Portfolio Management
    - Stock News Integration
    - Email Alerts
    """)

    st.success("✅ Project Developed Successfully using Python, Streamlit & Machine Learning.")
    st.subheader("📋 Dataset Preview")

    st.markdown("---")

    st.markdown(
        """
         <div style='text-align:center'>
            <h3>📈 AI Stock Market Forecast Prediction</h3>
            <p>Developed using Python, Streamlit, Machine Learning & Yahoo Finance</p>
            <p>👨‍💻 Developed by: <b>Shagun Gupta</b></p>
            <p>© 2026 All Rights Reserved</p>
        </div>
        """,
        unsafe_allow_html=True
)

    st.subheader("🎯 Key Features")

    st.markdown("""
✅ Interactive Dashboard

✅ Company Information

✅ Closing Price Analysis

✅ Moving Average Analysis

✅ Candlestick Chart

✅ Volume Analysis

✅ Daily Return Analysis

✅ Historical Data Download

✅ Machine Learning Prediction

✅ Tomorrow Price Prediction

✅ Investment Calculator

✅ Buy / Sell Recommendation

✅ Risk Meter

✅ Performance Statistics
""")

    st.dataframe(stock.tail(10), use_container_width=True)
    csv = stock.to_csv(index=False).encode("utf-8")

st.subheader("📊 Statistical Summary")

st.dataframe(stock.describe())
stock["Daily Return"] = stock["Close"].pct_change() * 100
st.subheader("📈 Daily Return")

return_fig = go.Figure()

return_fig.add_trace(
    go.Scatter(
        x=stock.index,
        y=stock["Daily Return"],
        mode="lines",
        name="Daily Return",
        line=dict(color="purple")
    )
)

return_fig.update_layout(
    template="plotly_dark",
    title="Daily Return (%)",
    xaxis_title="Date",
    yaxis_title="Return (%)",
    height=500
)

st.plotly_chart(
    return_fig,
    use_container_width=True,
    key="daily_return"
)
st.subheader("⚠️ Risk Level")

volatility = stock["Daily Return"].std()

if volatility < 1:
    st.success("🟢 Low Risk")

elif volatility < 2:
    st.warning("🟡 Medium Risk")

else:
    st.error("🔴 High Risk")

    with st.spinner("Loading Stock Data..."):

        stock = yf.download(
        stock_symbol,
        start=start_date,
        end=end_date
    )
    st.success("✅ Stock Data Loaded Successfully!")