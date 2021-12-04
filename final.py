import streamlit as st
import pandas as pd
import altair as alt
from pandas_datareader import data as pdr
from sklearn.cluster import KMeans
import sklearn
from sklearn.model_selection import train_test_split

st.markdown("[Nhi Truong](https://github.com/nhiphanthuctruong/FinalProject)")

st.title("Build Your Portfolio")
st.subheader("Cluster analysis")
st.markdown("* Cluster analysis is a technique used to group sets of objects that share similar characteristics. ")
st.markdown("* Investors will use cluster analysis to develop a cluster trading approach that helps them build a diversified portfolio that can reduce the risk for their portfolio.")
st.markdown("* Data using are top 10 stocks from Yahoo Finance")


# List name top 10 stock 
name = ['IRBT','AAPL','BB','GME','BBBY','BIO','AMZN','TSLA','NVDA','MSFT']
        
# Links specifying datasets from Yahoofinance for the 10 stock above
# This will be slow because it wil get data directly from Yahoofinace for each stock (means 10 different datasets)
if "data" in st.session_state:
    df1 = st.session_state["data"]
else:
    df1=pdr.DataReader(name,data_source='yahoo',start='2018-1-1')['Adj Close']
    st.session_state["data"] = df1



# Convert data to probabilities for ease of comparison between the stocks (analyze & clean the data)
df1 = df1.apply(pd.to_numeric)
daily = df1/df1.shift(1)-1 
daily = daily.dropna()
# cluster depend on the mean and standard deviation of stocks
df2 = daily.mean()
df2 = pd.DataFrame(df2,columns = {"Mean"})
df2["std"]=daily.std()

# Using scikit-learn
# apply KMeans to get cluster
kmeans = KMeans(3)
kmeans.fit(df2)
kmeans.predict(df2)
df2["cluster"] = kmeans.predict(df2)

# Show the cluster
st.markdown("Divide the 10 stocks into 3 different groups.")
value = st.radio("Pick a group",[0,1,2])
df3 = df2[df2['cluster'] == value]
st.write(df3)

# graph cluster
g = alt.Chart(df2).mark_circle().encode(
    x = "Mean",
    y = "std",
    tooltip = ["Mean","std"],
    color = "cluster:N"
)
st.altair_chart(g)
st.markdown("This graph divides top 10 stocks into different groups, so we recommend you choose the tickers in different groups for diversification which can reduce the risk for your portfolio.")




# Choice your ticker from 10 stock give above
st.subheader("Choose your ticker")
choice = st.multiselect("*If your tickers not appear in the graph below please delete and pick them again", name)


try:
    tickers = [choice]
    sec_data=pd.DataFrame()
    for t in tickers: 
        sec_data[t]=pdr.DataReader(t,data_source='yahoo',start= '2011-1-1')['Adj Close']
    
    #Use line chart to graph daily price of the stocks your picked   
    st.line_chart(sec_data)
    st.markdown("This graph shows the daily price of your portfolio to easy compare the price between the tickers.")
    
    #choose start date
    st.subheader("Summary")
    st.markdown("Choose start date")
    start_date = st.date_input('*If the table does not change, please pick start date again or reload the page.' )
    #st.markdown("With the start date, you can see the tickets in specific time up today to deeply how the ticker move in some event. ")
    st.markdown("With the start date, you can see the tickets in specific time up today to deeply understand how the ticker move in some event")
    #Normalize data to between 0 and 1
    sec_data1 = pdr.DataReader(choice ,data_source='yahoo',start= start_date)['Adj Close']
    daily_return = sec_data1 /sec_data1.shift(1)-1 
    daily_return = daily_return[daily_return.notna().all(axis = 1)].copy()

except:
    st.markdown("Please choose your tickers")

st.markdown("The explanation for summary table")
st.markdown("* Count: Total number dates of your have selected")
st.markdown("* Mean: Expect rate of return")
st.markdown("* Std: Standard deviation measure risk in the stock market")
st.markdown("* Min: Minimum rate of return per day")
st.markdown("* Max: Maximum rate of return per day")


# symmary stock
try:
    
    # Drop some row unnecessary in "describe"
    st.table(daily_return.describe().drop(["25%","50%","75%"]))

    st.subheader("Correlation matrix.")
    st.markdown("Correlation matrix gives out the correlation between any two stocks in a portfolio. If it close to ")
    st.markdown("* -1 indicates a perfect negative linear correlation between two tickers.")
    st.markdown("* 0 indicates no linear correlation between two variables.")
    st.markdown("* 1 indicates a perfect positive linear correlation between two tickers.")
   
    # Show Correlation matrix
    corr_matrix=daily_return.corr()
    st.table(corr_matrix)
except:
    st.markdown("Please choose your date")


# Predit stock price using 
st.subheader("Build Linear Regression to Predict Stock Price.")
st.markdown("The data for the predictor variable: Average return in last 2, 7, 30, and 356 dates")
st.markdown("The target variable: Adj Close Price")
st.markdown("Use train_test_split to split dataset to 80% tranin, 20% test")


try:
    #choose 1 stock form your portfolio
    value1 = st.radio("Pick a ticker you want to predict price",choice)
    df4 = pdr.DataReader(value1 ,data_source='yahoo',start='2012-1-1')
    # get average return last 2 dates
    df4['D_7'] = df4['Adj Close'].shift(1).rolling(window=7).mean()
    df4['D_2'] = df4['Adj Close'].shift(1).rolling(window=2).mean() 
    df4['D_30']= df4['Adj Close'].shift(1).rolling(window=30).mean()
    df4['D_365']= df4['Adj Close'].shift(1).rolling(window=365).mean()
    
    df4= df4.dropna() 
    X = df4[['D_2','D_7','D_30','D_365']] 
    y = df4['Adj Close']
    
    # use train_test_split Not in Math 10 to split to 80% tranin, 20% test
    train, test = train_test_split(df4, test_size=0.20)
    
    X_train = train[['D_2','D_7','D_30','D_365']]
    y_train = train["Adj Close"]
    X_test = test[['D_2','D_7','D_30','D_365']] 
    y_test = test["Adj Close"]
    
    
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    predicted_price = model.predict(X_test)
    
    # Displace our Linear Regression
    slope = model.coef_
    intercept = model.intercept_
    
    st.subheader("Our Linear Regression: ")
    st.markdown("---")
    st.write((f'Price Tomorrow = ({round(slope[0], 2)}) Two Dates + ({round(slope[1], 2)}) Seven Dates + ({round(slope[2], 2)}) Thirty Dates + ({round(slope[3], 2)}) One Year + ({round(intercept, 2)})'))
    st.markdown("---")
    
    predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['Predicted Price']) 
    predicted_price["Actual Price"] = y_test
    
    
    st.subheader("Test our model")
    # Use Altair to graph Actual Price and Predicted Price
    df5 = predicted_price
    df5["Actual Price"] = y_test
    df5 = df5.reset_index() # Reset index column to get date column
    
    chart1 = alt.Chart(df5).mark_line(color = 'red').encode(
        x = 'Date',
        y = 'Predicted Price',
        tooltip = ['Predicted Price','Date']
    ).properties(
        width = 700,
        height = 400
    )

    chart2 = alt.Chart(df5).mark_line(color = 'blue').encode(
        x = 'Date',
        y = 'Actual Price',
        tooltip = ['Actual Price','Date']
    ).properties(
        width = 700,
        height = 400
    )   
    st.altair_chart(chart2 + chart1)
    st.text("*Red line: Predicted Price & Blue Line: Actual Price ")

    
    st.markdown("This Graph tests our Linear Regression to show the difference in Actual Price and Predicted Price")
    
    
    st.subheader("What's the price tomorrow?")
    st.markdown("By clicking this button we will collect  the data and use the Linear Regression above to predict stock price tomorrow.")
    
    # Caculate price tomorrow from our Linear Regression Function
    df4 = pdr.DataReader(value1 ,data_source='yahoo',start='2012-1-1')['Adj Close']
    price = slope[0]*df4.tail(2).mean()+slope[1]*df4.tail(7).mean()+slope[2]*df4.tail(30).mean()+slope[3]*df4.tail(365).mean()+intercept
    price = round(price,3)
    if st.button("Predict Price Tomorrow"):
        st.write(price)
        st.balloons()

except:
    st.markdown("")
    
st.markdown("[Reference](https://medium.com/analytics-vidhya/using-linear-regression-to-predict-aapl-apple-stock-prices-in-python-1a629fbea15b)")
