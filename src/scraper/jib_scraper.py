import requests
from bs4 import BeautifulSoup
import re
import time
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("jib_scraper")

DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'th,en-US;q=0.9,en;q=0.8',
}

class JibRamScraper:
    """
    Scraper for live RAM listings and prices from JIB Computer Group.
    """
    CATEGORY_URLS = [
        # Desktop PC RAM
        {"category": "Desktop RAM", "url": "https://www.jib.co.th/web/product/product_list/2/53/{offset}"},
        # Laptop / Notebook RAM
        {"category": "Notebook RAM", "url": "https://www.jib.co.th/web/product/product_list/2/30/{offset}"},
    ]

    def __init__(self, request_delay: float = 0.5):
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def scrape_category(self, cat_info: Dict[str, str], max_pages: int = 15) -> List[Dict[str, Any]]:
        category_name = cat_info["category"]
        base_url_template = cat_info["url"]
        logger.info(f"Starting crawl for category: {category_name}")
        
        products = {}
        offset = 0
        page_num = 1

        while page_num <= max_pages:
            url = base_url_template.format(offset=offset)
            try:
                logger.info(f"Fetching {category_name} (Page {page_num}, Offset {offset})...")
                response = self.session.get(url, timeout=12)
                if response.status_code != 200:
                    logger.warning(f"Got status code {response.status_code} at {url}")
                    break

                soup = BeautifulSoup(response.text, "html.parser")
                cards = soup.find_all("div", class_="divboxpro")
                if not cards:
                    logger.info(f"No more product cards found at offset {offset}. Done.")
                    break

                new_items_in_page = 0
                for card in cards:
                    promo = card.find("span", class_="promo_name")
                    if not promo:
                        continue
                    
                    title = promo.get_text(strip=True)
                    a_link = promo.find_parent("a")
                    href = a_link["href"] if (a_link and a_link.has_attr("href")) else ""

                    card_text = card.get_text(" ", strip=True)
                    price_match = re.search(r"([\d,]+)\s*\.-", card_text)
                    if not price_match:
                        continue
                    
                    price = int(price_match.group(1).replace(",", ""))
                    item_id = href or title
                    if item_id not in products and price > 0:
                        products[item_id] = {
                            "title": title,
                            "price": price,
                            "url": href,
                            "category": category_name,
                            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }
                        new_items_in_page += 1

                logger.info(f"Page {page_num}: Found {len(cards)} cards, added {new_items_in_page} new items.")
                if new_items_in_page == 0:
                    logger.info("No new unique items on this page. Reached end of list.")
                    break

                offset += 50
                page_num += 1
                time.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                break

        logger.info(f"Finished category '{category_name}': {len(products)} unique items found.")
        return list(products.values())

    def scrape_all(self) -> List[Dict[str, Any]]:
        all_results = []
        for cat in self.CATEGORY_URLS:
            items = self.scrape_category(cat)
            all_results.extend(items)
        logger.info(f"Total products scraped across all categories: {len(all_results)}")
        return all_results
