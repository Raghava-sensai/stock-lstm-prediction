import pandas as pd
import os

# List of CSV files to merge
csv_files = [
    r'.\Data Extraction\NewsAPI\news_sentiment_analysis.csv',
    r'.\Data Extraction\WebScrapping\investing.com\news_sentiment_analysis.csv',
    r'.\Data Extraction\WebScrapping\investing.com\news_sentiment_analysis.csv',
    r'.\Data Extraction\WebScrapping\Reddit\r-news\news_sentiment_analysis.csv',
    r'.\Data Extraction\WebScrapping\Reddit\r-worldnews\news_sentiment_analysis.csv'
]

# Output file path
output_file = r'.\final_news_sentiment_analysis.csv'

# DataFrame to hold merged data
merged_df = pd.DataFrame()

# Read and merge each CSV file
for file in csv_files:
    if os.path.exists(file):  # Check if the file exists
        try:
            df = pd.read_csv(file)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    else:
        print(f"File not found: {file}")

# Sort by 'Company Name'
if 'Company Name' in merged_df.columns:
    merged_df.sort_values(by='Company Name', inplace=True)

# Save the merged DataFrame to a CSV
merged_df.to_csv(output_file, index=False)

print(f"Merged and sorted CSV saved to {output_file}")
