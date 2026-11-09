"""
Web Scraper & HTML Page Document Ingestion Engine.
Extracts clean main-body text while stripping scripts, styles, and navigation bars.
"""
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any, Tuple
from backend.app.core.logger import logger
from backend.app.core.exceptions import DocumentIngestionException

class WebLoader:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def fetch_url(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Async fetch and parse clean main text from a target web URL.
        """
        headers = {
            "User-Agent": "VeriMind-AI-Bot/2.2 (+https://github.com/verimind-ai/verimind)"
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                html_content = response.text

            soup = BeautifulSoup(html_content, "html.parser")

            # Strip script, style, header, footer elements
            for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                element.decompose()

            # Extract title and body text
            title = soup.title.string.strip() if soup.title and soup.title.string else url
            body_text = soup.get_text(separator="\n", strip=True)

            metadata = {
                "source": url,
                "title": title,
                "file_type": "webpage",
                "http_status": response.status_code,
                "total_char_length": len(body_text)
            }

            return body_text, metadata

        except Exception as e:
            logger.error(f"Web scraper error fetching {url}: {str(e)}")
            raise DocumentIngestionException(f"Failed to load web page: {str(e)}")

web_loader = WebLoader()