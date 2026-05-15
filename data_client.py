import requests

BASE_URL = "https://jsonplaceholder.typicode.com"

class DataClient:
    @staticmethod
    def fetch_posts(limit: int = 10) -> list[dict]:
        response = requests.get(f"{BASE_URL}/posts", timeout=10)
        response.raise_for_status()
        return response.json()[:limit]