from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time, csv
from urllib.parse import quote_plus

# ─── CONFIG ────────────────────────────────────────────────────────────────
KEYWORD       = "maybank marathon 2024"
SEARCH_URL    = f"https://www.tiktok.com/search?q={quote_plus(KEYWORD)}"
OUTPUT_DIR    = "./"
SCROLL_ITER   = 10      # scroll count for search results
PAGE_PAUSE    = 3      # seconds between page scrolls
COMMENT_PAUSE = 2      # seconds between comment scrolls
MAX_SCROLL    = 5     # max fractional scrolls to reveal comments
SCROLL_DELTA  = 600    # pixels per scroll step
# ────────────────────────────────────────────────────────────────────────────

def setup_driver():
    opts = webdriver.ChromeOptions()
    # opts.add_argument("--headless")  # uncomment to run headless
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )


def fetch_video_urls(driver):
    driver.get(SEARCH_URL)
    time.sleep(PAGE_PAUSE * 2)
    for _ in range(SCROLL_ITER):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(PAGE_PAUSE)
    anchors = driver.find_elements(By.TAG_NAME, "a")
    urls = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and "/video/" in href:
            urls.append(href.split('?')[0])
    seen, result = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); result.append(u)
    return result


def scrape_comments(driver, url):
    driver.get(url)
    time.sleep(PAGE_PAUSE * 2)
    # Scroll in small increments to load comments
    scroll_script = "window.scrollBy(0, arguments[0]);"
    for _ in range(MAX_SCROLL):
        driver.execute_script(scroll_script, SCROLL_DELTA)
        time.sleep(COMMENT_PAUSE)
    # After incremental scrolls, collect comment texts
    elems = driver.find_elements(
        By.CSS_SELECTOR,
        "span[data-e2e='comment-level-1'] p.TUXText"
    )
    comments = [{'text': el.text.strip().replace('\n',' ')}
                for el in elems if el.text.strip()]
    return comments


def main():
    driver = setup_driver()
    urls = fetch_video_urls(driver)
    print(f"Found {len(urls)} videos to scrape comments for.")
    for url in urls:
        comments = scrape_comments(driver, url)
        vid = url.rstrip('/').split('/')[-1]
        fn = f"{OUTPUT_DIR}comments_{vid}.csv"
        with open(fn, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text'])
            writer.writeheader(); writer.writerows(comments)
        print(f"Saved {len(comments)} comments for video {vid}.")
        time.sleep(PAGE_PAUSE)
    driver.quit()

if __name__ == '__main__':
    main()

