import os
import pandas as pd
from datetime import datetime

# Read the input CSV with company details
company_data = pd.read_csv(r'.\Info.csv')  # Replace with your CSV file path

# Define paths for stock data and news data
stock_data_folder = r'.\NIFTY 50'
news_data_folder = r'.\r-worldnews\Data'

# Output files
output_file = r'.\r-worldnews\news_sentiment_analysis.csv'
debug_log_file = r'.\r-worldnews\debug_log.txt'

# Prepare the output CSV
output_columns = ['Company Name', 'Symbol', 'Headline', 'Publish Date', 'Sentiment']
output_df = pd.DataFrame(columns=output_columns)

# Function to calculate sentiment based on percentage change
def calculate_sentiment_with_delta(stock_data, publish_date):
    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')  # Handle invalid values
    next_available_data = stock_data[stock_data['Date'] > publish_date]
    
    if len(next_available_data) < 2:
        return 'Neutral'
    
    next_day_price = next_available_data.iloc[0]['Close']
    day_after_next_price = next_available_data.iloc[1]['Close']
    
    if pd.isna(next_day_price) or pd.isna(day_after_next_price):
        return 'Neutral'
    
    percentage_change = ((day_after_next_price - next_day_price) / next_day_price) * 100
    if abs(percentage_change) <= 1:
        return 'Neutral'
    elif percentage_change > 1:
        return 'Positive'
    else:
        return 'Negative'

# Open debug log file
with open(debug_log_file, 'w', encoding='utf-8') as debug_log:
    for _, row in company_data.iterrows():
        company_name = row['Company Name']
        symbol = row['Symbol']
        
        stock_data_file = os.path.join(stock_data_folder, f"{symbol}.csv")
        news_data_file = os.path.join(news_data_folder, f"{symbol}.csv")
        
        debug_log.write(f"Processing company: {company_name}, symbol: {symbol}\n")
        
        # Load stock data
        if not os.path.exists(stock_data_file):
            debug_log.write(f"Stock data file for {symbol} not found. Skipping.\n")
            continue
        
        try:
            stock_data = pd.read_csv(stock_data_file, parse_dates=['Date'])
            stock_data['Date'] = stock_data['Date'].dt.date
        except Exception as e:
            debug_log.write(f"Error loading stock data for {symbol}: {e}\n")
            continue

        # Load news data
        if not os.path.exists(news_data_file):
            debug_log.write(f"News data file for {symbol} not found. Skipping.\n")
            continue
        
        try:
            news_data = pd.read_csv(news_data_file, parse_dates=['Date'])
            news_data['Date'] = news_data['Date'].dt.date
        except Exception as e:
            debug_log.write(f"Error loading news data for {symbol}: {e}\n")
            continue
        
        # Process each news headline
        for _, article in news_data.iterrows():
            headline = article['Headline']
            publish_date = article['Date']
            
            debug_log.write(f"Processing article: '{headline}', published on {publish_date}\n")
            
            if publish_date in stock_data['Date'].values:
                sentiment = calculate_sentiment_with_delta(stock_data, publish_date)
            else:
                debug_log.write(f"No stock data available for publish date {publish_date}. Skipping.\n")
                sentiment = 'Neutral'
            
            output_df = pd.concat([
                output_df,
                pd.DataFrame([[company_name, symbol, headline, publish_date, sentiment]], columns=output_columns)
            ], ignore_index=True)
    
    debug_log.write("Processing complete.\n")

# Save results to CSV
output_df.to_csv(output_file, index=False)
print(f"Sentiment analysis results saved to {output_file}")
print(f"Debug log saved to {debug_log_file}")
