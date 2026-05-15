import requests

class DataClient:
    @staticmethod
    def fetch_posts(limit=10):
        try:
            response = requests.get("https://jsonplaceholder.typicode.com/posts")
            return response.json()[:limit]
        except Exception as e:
            print(f"Data error: {e}")
            return []