from data_extract.interfaces import DataExtractInterface
import logging
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
logging.getLogger("icrawler").setLevel(logging.ERROR)

class Extractor(DataExtractInterface):
    def __init__(self):
        self._name = "Extract"
        self._info = "Extract images from google or bing"

    def extract_data(self, query: str, max_num: int, search_engine: list, path: str) -> str:
        """
        Extract images from specified search engines and save them to the given path.

        Args:
            query (str): The search keyword.
            max_num (int): The maximum number of images to download.
            search_engines (list): List of search engines to use (supports 'google' and 'bing').
            path (str): Directory path to save the images.

        Returns:
            str: Success message after extraction.
        """
        # Dictionary mapping search engines to their respective crawlers
        crawlers = {
            "google": GoogleImageCrawler,
            "bing": BingImageCrawler
        }

        # Extract images from specified engines
        for engine in search_engine:
            if engine not in crawlers:
                raise ValueError(f"Unsupported search engine: {engine}")

            storage_dir = f"{path}/{query}" if engine == "google" else f"{path}/{query}_bing"
            crawler = crawlers[engine](storage={'root_dir': storage_dir})
            crawler.crawl(keyword=query, max_num=max_num)

        return "Data extracted successfully"