import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import random

# Set up Selenium WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Optional: Run in headless mode
# options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
driver = webdriver.Chrome(service=service, options=options)

# Set the output folder where CSV files will be saved
output_folder = r'.\Data Extraction\WebScrapping\Reddit\r-news\Data'
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file containing company details
with open(r'.\Info.csv', mode='r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        time.sleep(random.uniform(1, 3)) 
        company_name = row['Company Name']
        industry = row['Industry']
        symbol = row['Symbol']
        keyword = row['Keyword']
        
        # URL of the Reddit search page, using the 'Keyword' from the CSV
        url = f'https://www.reddit.com/r/news/search/?q={keyword}'

        try:
            # Load the search page
            driver.get(url)
            time.sleep(5)  # Allow time for the page to load

            # Wait for posts to load (first attempt)
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="post-title-text"]'))
                )
                posts_loaded = True
            except:
                print(f"Posts for {company_name} ({symbol}) could not be loaded. Skipping.")
                posts_loaded = False

            if posts_loaded:
                # Scroll the page to load more posts
                prev_height = driver.execute_script("return document.body.scrollHeight")  # Get initial scroll height
                max_scroll_attempts = 50
                for attempt in range(max_scroll_attempts):
                    driver.execute_script("window.scrollBy(0, 1000);")  # Scroll down 1000px
                    time.sleep(2)  # Wait for posts to load

                    # Check if new content has been loaded
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == prev_height:  # No change in height
                        print(f"No more posts loaded after scrolling for {company_name} ({symbol}). Moving to next query.")
                        break
                    prev_height = new_height  # Update the height for the next comparison

                # Get the page source after scrolling
                page_source = driver.page_source

                # Parse the page source with BeautifulSoup
                soup = BeautifulSoup(page_source, 'html.parser')

                # Find all post containers
                posts = soup.find_all('a', {'data-testid': 'post-title-text'})

                # Store results in a list
                post_data = []

                # Loop through posts and extract data
                for post in posts:
                    headline = post.text.strip()

                    # Find the timestamp in the parent container
                    parent_div = post.find_parent('div')  # Get the parent div
                    timestamp_elem = parent_div.find('faceplate-timeago') if parent_div else None
                    timestamp = timestamp_elem['ts'].split('T')[0] if timestamp_elem else 'No date'

                    post_data.append([headline, timestamp])

                # Write the results to a CSV file named after the symbol (e.g., SYMBOL.csv)
                output_file = os.path.join(output_folder, f'{symbol}.csv')
                with open(output_file, mode='w', newline='', encoding='utf-8') as result_file:
                    writer = csv.writer(result_file)
                    writer.writerow(['Headline', 'Date'])  # Write header
                    writer.writerows(post_data)  # Write post data

                print(f"Results for {company_name} ({symbol}) saved in {output_file}")
            else:
                continue

        except Exception as e:
            print(f"An error occurred while processing {company_name} ({symbol}): {e}")

# Close the browser
driver.quit()
