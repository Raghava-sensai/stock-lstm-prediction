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
from selenium.webdriver.common.action_chains import ActionChains

# Set up Selenium WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument("--ignore-certificate-errors")
options.add_argument('--ignore-ssl-errors=yes')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

# Set the output folder where CSV files will be saved
output_folder = r'.\investing.com\Data'
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file containing company details
with open(r'.\Info.csv', mode='r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        time.sleep(random.uniform(1, 2))
        company_name = row['Company Name']
        industry = row['Industry']
        symbol = row['Symbol']
        Link = row['Link']
        L_limit = int(row['L_limit'])  # Read L_limit from CSV
        
        post_data = []  # Store post data for each company
        
        # Loop through pages
        for page in range(1, L_limit + 1):
            url = f'https://www.investing.com/equities/{Link}/{page}'
            print(f"Processing {company_name} ({symbol}), Page: {page}")
            
            try:
                # Load the page
                driver.get(url)
                time.sleep(1)  # Allow time for the page to load
                
                # Handle the verification page
                try:
                    # Wait for a selector specific to the challenge page (e.g., a JavaScript element or button)
                    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "cf-chl-consent")))  # Adjust this selector based on the page structure
                    print("Bypassing Cloudflare challenge...")
                    # Perform a simple action to bypass (can adjust according to the challenge page)
                    action = ActionChains(driver)
                    action.move_to_element(driver.find_element(By.TAG_NAME, "body")).perform()
                    time.sleep(10)  # Wait for challenge to pass
                except Exception as e:
                    print(f"No verification challenge on page {page}: {e}")
                
                # Parse the page with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Iterate through each news list
                news_lists = soup.find_all("ul", {"data-test": "news-list"})
                for news_list in news_lists:
                    # Find all article items inside the current news list
                    articles = news_list.find_all("article", {"data-test": "article-item"})
    
                    for article in articles:
                        # Extract the headline
                        headline_tag = article.find("a", {"data-test": "article-title-link"})
                        headline = headline_tag.text.strip() if headline_tag else 'No headline'

                        # Extract the date
                        time_tag = article.find('time', {'data-test': 'article-publish-date'})
                        date = time_tag['datetime'].split(' ')[0] if time_tag else 'No date'

                        # Append the data to the list
                        post_data.append([headline, date])
                    
            except Exception as e:
                print(f"An error occurred on page {page} for {company_name} ({symbol}): {e}")
                break  # Exit the loop if there's a major issue

            # Write the results to a CSV file named after the symbol (e.g., SYMBOL.csv)
            output_file = os.path.join(output_folder, f'{symbol}.csv')
            with open(output_file, mode='a', newline='', encoding='utf-8') as result_file:
                writer = csv.writer(result_file)
                if os.stat(output_file).st_size == 0:  # Check if file is empty, add header only if needed
                    writer.writerow(['Headline', 'Date'])
                writer.writerows(post_data)
            print(f"Results for {company_name} ({symbol}) saved in {output_file}")

# Close the browser
driver.quit()
