from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Selenium options
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run Chrome in headless mode

# Setup webdriver service
s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s, options=options)

# Navigate to the webpage
driver.get('https://www.target.com/p/east-bluff-woven-drawer-console-table-black-threshold-8482-designed-with-studio-mcgee/-/A-78140507')

# Wait for the page to load and reviews to be fetched
wait = WebDriverWait(driver, 20)  # Increase the timeout to 20 seconds

reviews = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'h-margin-t-default.h-text-md')))

# Loop through the reviews and print them
for review in reviews:
    print(review.text)

# Close the browser
driver.quit()
