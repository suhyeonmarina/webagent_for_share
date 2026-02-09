import traceback
from playwright.sync_api import sync_playwright
import os

print("DISPLAY:", os.environ.get('DISPLAY'))
print("Starting browser test...")

try:
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=False)
        print("Browser launched successfully!")

        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        print("Context created!")

        page = context.new_page()
        print("Page created!")

        print("Navigating to Amazon...")
        page.goto("https://www.amazon.com", timeout=30000)
        print("Page loaded successfully!")
        print("Current URL:", page.url)

        browser.close()
        print("Test completed successfully!")

except Exception as e:
    print("\n=== ERROR ===")
    print(traceback.format_exc())
