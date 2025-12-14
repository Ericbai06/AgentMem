import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 双库配置
    MEMOS_ORIGIN_API_KEY = os.getenv("MEMOS_ORIGIN_API_KEY")
    MEMOS_PROCESS_API_KEY = os.getenv("MEMOS_PROCESS_API_KEY")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL", "gpt-4o")
    
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "locomo10.json")
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")

    MAX_WORKERS_INGEST = 10
    MAX_WORKERS_SEARCH = 5

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)