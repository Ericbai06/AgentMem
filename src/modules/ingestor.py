import json
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from memos.api.client import MemOSClient
from .config import Config
from .prompts import FACT_EXTRACTION_PROMPT

class MemoryIngestor:
    def __init__(self):
        # 连接到 Process 库
        self.client = MemOSClient(api_key=Config.MEMOS_PROCESS_API_KEY)
        self.llm_client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
        self.data = self._load_data()

    def _load_data(self):
        with open(Config.DATA_PATH, "r") as f:
            return json.load(f)

    def extract_and_format_facts(self, user_name, text, date):
        # ... (保持之前的提取逻辑不变) ...
        try:
            prompt = FACT_EXTRACTION_PROMPT.format(user_name=user_name, date=date, conversation_text=text)
            response = self.llm_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception:
            return text

    def add_single_memory(self, user_id, content, timestamp):
        try:
            self.client.add_message(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                conversation_id=timestamp
            )
        except Exception:
            pass

    
    def process_conversation_chunk(self, user_id, user_name, messages, timestamp):
        if not messages: return
        full_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        extracted_facts = self.extract_and_format_facts(user_name, full_text, timestamp)
        self.add_single_memory(user_id, extracted_facts, timestamp)

    def process_item(self, idx, item):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        id_a = f"{speaker_a}_{idx}"
        id_b = f"{speaker_b}_{idx}"

        for key in conversation:
            if key in ["speaker_a", "speaker_b"] or "_date_time" in key: continue
            date_key = key + "_date_time"
            if date_key not in conversation: continue
            timestamp = conversation[date_key]
            chats = conversation[key]

            msgs_for_a = [{"role": "user" if c["speaker"] == speaker_a else "other", "content": c["text"]} for c in chats]
            msgs_for_b = [{"role": "user" if c["speaker"] == speaker_b else "other", "content": c["text"]} for c in chats]

            t1 = threading.Thread(target=self.process_conversation_chunk, args=(id_a, speaker_a, msgs_for_a, timestamp))
            t2 = threading.Thread(target=self.process_conversation_chunk, args=(id_b, speaker_b, msgs_for_b, timestamp))
            t1.start(); t2.start(); t1.join(); t2.join()

    def run(self):
        print(f"🧠 Starting Fact Extraction (to Process DB)...")
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS_INGEST) as executor:
            futures = [executor.submit(self.process_item, idx, item) for idx, item in enumerate(self.data)]
            for _ in tqdm(futures, total=len(self.data), desc="Ingesting Facts"):
                _.result()
        print("✅ Fact Ingestion Complete.")

if __name__ == "__main__":
    ingestor = MemoryIngestor()
    ingestor.run()