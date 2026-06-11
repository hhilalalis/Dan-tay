import asyncio
import shutil
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def scrape_cases(search_query: str = "vergi", max_cases: int = 5) -> list[dict]:
    """
    Scrapes historical case law from karararama.danistay.gov.tr.
    Since the exact DOM structure may change, this function attempts to 
    navigate to the site, perform a search, and extract the text.
    
    Args:
        search_query: The search terms to use on the portal
        max_cases: Maximum number of cases to extract
    
    Returns:
        List of case dictionaries with url, query, and content
    """
    url = "https://karararama.danistay.gov.tr/"
    cases = []

    logger.info(f"Starting Playwright to scrape {url} for query '{search_query}' (max {max_cases} cases)")
    
    async with async_playwright() as p:
        system_chromium = shutil.which("chromium-browser") or shutil.which("chromium")
        launch_args = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if system_chromium:
            launch_args["executable_path"] = system_chromium
        browser = await p.chromium.launch(**launch_args)
        page = await browser.new_page()
        
        try:
            # Navigate to the website
            await page.goto(url, wait_until="networkidle")
            
            # Update selectors based on our research of the Danistay DOM
            search_input_selector = "input#andKelime"
            search_button_selector = "#kelimeAraVe button:has-text('Ara')"
            
            # Use jQuery to select the search type to reveal the input fields
            logger.info("Selecting search type 'andKelime'...")
            await page.evaluate("""() => {
                $('#searchType').val('andKelime').trigger('change');
                $('#searchType').trigger({
                    type: 'select2:select',
                    params: { data: { id: 'andKelime' } }
                });
            }""")
            
            # Wait for the input to become visible
            try:
                await page.wait_for_selector(search_input_selector, state="visible", timeout=5000)
                logger.info("Found search input, filling query...")
                await page.fill(search_input_selector, search_query)
                
                logger.info("Clicking search button...")
                await page.click(search_button_selector)
                
                # Wait for the results grid to load
                logger.info("Waiting for results...")
                await page.wait_for_timeout(3000) 
                
                try:
                    # Wait for a table cell to appear (indicating data loaded)
                    await page.wait_for_selector('#detayAramaSonuclar tbody tr td.sorting_1, #detayAramaSonuclar tbody tr:not(.odd) td', timeout=5000)
                except Exception:
                    logger.warning("Table data did not load in time. Proceeding anyway...")
                
                # Expand details if any
                if await page.locator('#detay').is_visible():
                    await page.click('#detay')
                    
            except Exception as e:
                logger.warning(f"Could not interact with search form correctly: {e}. Extracting visible text as fallback.")
            
            # Extract multiple cases by clicking on different rows
            rows = page.locator('#detayAramaSonuclar tbody tr')
            row_count = await rows.count()
            logger.info(f"Found {row_count} rows in results table")
            
            # Extract up to max_cases cases
            for i in range(min(max_cases, row_count)):
                try:
                    row = rows.nth(i)
                    logger.info(f"Clicking case row {i+1}/{min(max_cases, row_count)}...")
                    
                    async with page.expect_popup() as popup_info:
                        await row.click()
                    
                    popup = await popup_info.value
                    logger.info("Popup opened. Waiting for it to load...")
                    await popup.wait_for_load_state("networkidle")
                    
                    case_html = await popup.content()
                    case_soup = BeautifulSoup(case_html, "html.parser")
                    
                    # Extract text from the popup
                    text = case_soup.get_text(separator="\n")
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    cleaned_text = "\n".join(lines)
                    
                    if cleaned_text:
                        cases.append({
                            "url": popup.url,
                            "query": search_query,
                            "content": cleaned_text[:50000]  # Limit to 50k chars
                        })
                        logger.info(f"Successfully extracted {len(cleaned_text)} characters from case {i+1}")
                    else:
                        logger.warning(f"No text found in case document popup {i+1}")
                    
                    await popup.close()
                    
                    # Wait a bit before clicking the next row
                    await page.wait_for_timeout(500)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract case {i+1}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"An error occurred during scraping: {e}")
        finally:
            await browser.close()
            
    logger.info(f"Scraping complete. Extracted {len(cases)} case(s)")
    return cases

if __name__ == "__main__":
    # Test the scraper for one case
    cases = asyncio.run(scrape_cases(max_cases=1))
    for i, case in enumerate(cases):
        print(f"Case {i+1}: {case['content'][:200]}...")
