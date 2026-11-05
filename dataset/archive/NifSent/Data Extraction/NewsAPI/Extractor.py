import os
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='InputAPI Key')

# Read the input CSV with company details
company_data = pd.read_csv(r'.\Info.csv')  # Replace with your CSV file path

# Define the folder containing stock data files
stock_data_folder = r'.\NIFTY 50'  # Replace with the actual path

# Output files
output_file = 'news_sentiment_analysis.csv'
debug_log_file = 'debug_log.txt'

# Prepare the output CSV
output_columns = ['Company Name', 'Symbol', 'Headline', 'Publish Date', 'Sentiment']
output_df = pd.DataFrame(columns=output_columns)

# Function to calculate sentiment based on percentage change
def calculate_sentiment_with_delta(stock_data, publish_date):
    # Ensure 'Close' column is numeric
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')  # Coerce invalid values to NaN
    
    # Find the closest date after the publish date
    next_available_data = stock_data[stock_data['Date'] > publish_date]
    
    if len(next_available_data) < 2:
        # If there aren't two future data points, sentiment is neutral
        return 'Neutral'
    
    # Get prices for the next two available dates
    next_day_price = next_available_data.iloc[0]['Close']
    day_after_next_price = next_available_data.iloc[1]['Close']
    
    # Check if prices are valid (not NaN)
    if pd.isna(next_day_price) or pd.isna(day_after_next_price):
        return 'Neutral'
    
    # Calculate percentage change
    percentage_change = ((day_after_next_price - next_day_price) / next_day_price) * 100

    # Determine sentiment
    if abs(percentage_change) <= 1:
        return 'Neutral'
    elif percentage_change > 1:
        return 'Positive'
    else:
        return 'Negative'

# Open debug log file
with open(debug_log_file, 'w', encoding='utf-8') as debug_log:
    # Iterate through each company in the CSV
    for _, row in company_data.iterrows():
        company_name = row['Company Name']
        keyword = row['Keyword']
        symbol = row['Symbol']
        stock_data_file = os.path.join(stock_data_folder, f"{symbol}.csv")  # Path to stock data file
        
        debug_log.write(f"Processing company: {company_name}, symbol: {symbol}, keyword: {keyword}\n")
        
        # Check if stock data file exists
        if not os.path.exists(stock_data_file):
            debug_log.write(f"Stock data file for {symbol} not found. Skipping.\n")
            continue
        
        # Read stock data
        try:
            stock_data = pd.read_csv(stock_data_file, parse_dates=['Date'])
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date  # Normalize date format
            debug_log.write(f"Loaded stock data for {symbol}, total rows: {len(stock_data)}\n")
        except Exception as e:
            debug_log.write(f"Error loading stock data for {symbol}: {e}\n")
            continue
        
        # Fetch news articles for the company
        try:
            all_articles = newsapi.get_everything(q=keyword,
                                                  from_param='2024-10-25',
                                                  to='2024-11-25',
                                                  language='en',
                                                  sort_by='relevancy')
            articles = all_articles.get('articles', [])
            debug_log.write(f"Fetched {len(articles)} articles for keyword: {keyword}\n")
        except Exception as e:
            debug_log.write(f"Error fetching articles for {keyword}: {e}\n")
            continue
        
        for article in articles:
            headline = article['title']
            publish_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date()

            debug_log.write(f"Processing article: '{headline}', published on {publish_date}\n")
            
            # Check if publish date is in stock data
            if publish_date in stock_data['Date'].values:
                today_price = stock_data.loc[stock_data['Date'] == publish_date, 'Close'].values[0]
                sentiment = calculate_sentiment_with_delta(stock_data, publish_date)
                debug_log.write(f"Sentiment for article '{headline}' is {sentiment}\n")
            else:
                debug_log.write(f"No stock data available for publish date {publish_date}. Looking for the next available date.\n")
                sentiment = calculate_sentiment_with_delta(stock_data, publish_date)
                debug_log.write(f"Calculated sentiment for the next available date is {sentiment}\n")
            
            # Append result to output DataFrame
            output_df = pd.concat([
                output_df,
                pd.DataFrame([[company_name, symbol, headline, publish_date, sentiment]], columns=output_columns)
            ], ignore_index=True)
    
    debug_log.write("Processing complete.\n")

# Save results to CSV
output_df.to_csv(output_file, index=False)
print(f"Sentiment analysis results saved to {output_file}")
print(f"Debug log saved to {debug_log_file}")
