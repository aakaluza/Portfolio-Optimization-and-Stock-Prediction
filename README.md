# Portfolio Optimization and Stock Prediction
*This is the page for proposal submission. Will move off enterprise and onto personal once account approved for Student Development*
Members: Olaolu Dada | Eric Shavkin | Dom Fernando | Logan Drawdy | Alek Kaluza

## Introduction / Background
With new technologies, many financial institutions are adopting machine learning to mitigate human error in computation and, in turn, maximize profits while minimizing risk of loss creating investment portfolios. We will be using datasets on historical stock prices from Yahoo! Finance API to derive financial indicators such as cumulative return, portfolio risk metrics, and more for performance prediction and portfolio optimization (Wu, Wang, Wu, 2022).

## Problem Definition
Our goal is to learn about potential methods used to analyze and predict stock behavior. We plan to find gaps in current research, particularly the use of clustering to derive optimal investment portfolios. Markowitz’s mean-variance portfolio is today’s standard optimization technique but it cannot help individuals determine what investments to make from a large pool of stocks (Wu, Wang, Wu, 2022). We want to try and bridge this gap by combining financial computations such as mean-variance and Sharpe Ratio with unsupervised learning techniques. Furthermore, we plan to use supervised learning techniques to predict portfolio performance over time, looking at areas of higher risk of loss and attempting to find ways to hedge this loss. The problem with many common predictive models being applied to the market is trying to fit a linear model to a non-linear reality.

## Data Collection and Feature Creation
After doing some research, we found and selected features of stocks that we felt would help us determine whether or not the stock would be a good investment for portfolio creation. These features are average return, volatility, beta, CAPM, Sharpe Ratio, average volume, EPS, and RSI and they are defined as such:
* Average Return: the average percent return of adjusted returns over a series of time
* Volatility: the standard deviation of percent returns over a series of time
* Beta: the covariance of the percent returns divided by the overall variance of the market
* CAPM: the risk-free rate of the market (defined in the US as the percentage return on a 10-year US treasury bond with 0 risk) + beta * (the market average return - risk-free rate)
* Sharpe Ratio: (CAPM - risk-free rate) / volatility
* Average Volume: the amount of equity shares on average traded daily over the course of three years
* EPS: earnings per share, a measure of a company's profitability
* RSI: Relative Strength Index, 100 + (100/(1 + the absolute value of the average gain / average loss over a 14 day period)). It is a determination of a bullish or bearish market.

We selected these features because they give us a good sense of how an equity is doing relative to the rest of the market over a long span of time. They also tell us the risk associated with an investment and whether it is an opportune time to invest given the performance over a smaller period of time.

To collect our data, we compiled CSV viles of every equity listed on the NASDAQ, NYSE, LSE, and SGI, using the yahoo_fin API to retrieve historical data from November 3rd, 2018 to November 3rd, 2023. We chose this range because it gives us a good idea of how a stock performed pre and post the pandemic. As we know, the pandemic had significant effects on market trends, particularly regarding a bearish trend with significant volatility in pharmaceutical and biotechnology investments. We planned on using the S&P 500 ETF to show market trends for the NASDAQ and NYSE, the FTSE 100 ETF for the LSE, and the STI 100 ETF for the SGI. However, when we initially collected data, we found problems in translating the risk of investments in foreign markets to US markets, specifically regarding the risk-free rate in other markets (which we use in calculation of our features). Therefore, we narrowed the scope of our study to only focus on US markets, only utilizing data from the NYSE, NASDAQ, and S&P 500.

Once our data was collected, we found that we had many null values for the EPS feature so we dropped it as we concluded there was not enough data to consider utilizing it for our model. Average Volume was also dropped from the list of features because we felt that it provided less valuable information than other selected features and yielded significant variance that we believed would affect the output of the model significantly. After all of this, we then had data that would suffice in identifying stocks to use in portfolio creation, structured as such in the table of 6047 stocks seen below:

### INSERT TABLE HERE
(Only NASDAQ is shown because of the way data is sorted)

The data was cleaned of any null values that it created (i.e. a stock with insufficient data). Next, we had to remove outliers from the data. These outliers consisted of stocks such as ACCD in the table above, which had significantly larger average return and volatility values. These trends can be associated with the large influx of investment in small and volatile bio-technology and pharmaceutical companies during the pandemic, which have now stabilized at significantly lower returns. In addition, this includes stocks with very randomnly large, and almost instantaneous increases in return and traded volume, likely due to the nature of high frequency trading bots exploiting the market inefficiencies. Furthermore, we wanted to eliminate stocks such as ACABU in the table above, having an RSI of 0, which is an indication of continuous 14 day loss with no gain, or stagnation of price return as a whole. To do this, we calculated the z-scores for each data point at each feature and eliminated any data point that had a feature with a z-score feature greater than 3 (meaning that the point was farther than 3 standard deviations from the average of that particular feature for that data set), classifying it as an outlier assuming a normal distribution of the data. 

With all of this cleaning complete, we were left with 6273 stocks to use for our model.

## Methods
In order to create an optimized portfolio, we will use K-Means clustering to determine what to invest in and how much to invest. Then, we will attempt to use mathematical models from the field of chaos theory, the study of dynamic systems whose ‘randomness’ is primarily attributed to small changes in initial conditions (Sahni, 2018). If we succeed in combining these mathematical models with supervised learning techniques, including a radial bias function and a long short term memory network, we hope to predict portfolio performance, find potential risks of loss, and determine hedging techniques to mitigate that loss (Li, Deng, 2007; Moghar, Hamiche, 2020).

## Potential Results and Discussions
Our results we expect from dataset analysis may depend on several factors that we will narrow down as we do further research. Considering scope we ask: How are back do we draw data from? Do we draw from short-term or long-term trends? The number of data points our models should be able to handle will depend on the domain problem complexity and the models addressing it. Will we need to perform dimensionality reduction to simplify features? We need to source our data from trusted sources to ensure its quality and availability so that we may develop our models. Moreover, we must clean our dataset of errors, duplicates, and preprocess as needed before applying our models. Ultimately, our objective is defined by what our model can accomplish and how it will benefit users looking to optimize their portfolios. These considerations will become more clear to us as we delve deeper into the project.
Some common metrics we will use to evaluate our prediction model are:
1. Distinct prediction accuracy
2. Precision and Recall
3. Sharpe Ratio (Risk-Adjusted Return Measure)
4. Mean Absolute Error and Mean Squared Error

## References
Dingming Wu, Xiaolong Wang, Shaocong Wu, “Construction of stock portfolios based on k-means clustering of continuous trend features”, Knowledge-Based Systems, Volume 252, 2022, 109358, ISSN 0950-7051.

R. Sahni, "Analysis of Stock Market Behaviour by Applying Chaos Theory," 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT), Bengaluru, India, 2018, pp. 1-4, doi: 10.1109/ICCCNT.2018.8494103.

X. Li and Z. Deng, "A Machine Learning Approach to Predict Turning Points for Chaotic Financial Time Series," 19th IEEE International Conference on Tools with Artificial Intelligence(ICTAI 2007), Patras, Greece, 2007, pp. 331-335, doi: 10.1109/ICTAI.2007.105.

Adil Moghar, Mhamed Hamiche, “Stock Market Prediction Using LSTM Recurrent Neural Network”, Procedia Computer Science, Volume 170, 2020, pp. 1168-1173, ISSN 1877-0509, https://doi.org/10.1016/j.procs.2020.03.049.

## Project Timeline
Here is a link to our project timeline as a Gantt Chart. If there is an issue with viewing the sheet please let us know but anyone should be able to view with just the link.

[Project Time Table](https://docs.google.com/spreadsheets/d/1u52pXjQm2g6DSqXZazK3U_52LHC-w85W/edit?usp=sharing&ouid=103164204269219948791&rtpof=true&sd=true)

## Contribution Table
Alek Kaluza: GitHub Setup, Video Recording<br>
Logan Drawdy: Introduction/Background, Problem Definition<br>
Eric Shavkin: Problem Definition, Methods<br>
Olaolu Dada: Methods, Project Timeline<br>
Dom Fernando: Potential Results and Discussion

## Checkpoint
**Midterm Report**
- Fully cleaned dataset
- Expect proper K-Means Clustering of data
- Provide visuals of clustering
- Begin supervised learning portion

**Final Report**
- Have supervised learning complete
- Provide more visuals to demonstrate stock prediction
