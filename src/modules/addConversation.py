import json
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from memos.api.client import MemOSClient
from .config import Config

class RawConversationAdder:
    def __init__(self, batch_size=5):
        # 关键修改：这里使用的是 MEMOS_ORIGIN_API_KEY (原始数据库)
        self.client = MemOSClient(
            api_key=Config.MEMOS_ORIGIN_API_KEY
        )
        self.batch_size = batch_size
        self.data_path = Config.DATA_PATH
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            return json.load(f)

    def add_raw_memory(self, user_id, messages, timestamp, retries=3):
        """
        调用 API 存储。
        timestamp 直接作为 conversation_id 传入，不写入 content 文本中。
        """
        for attempt in range(retries):
            try:
                # MemOS uses conversation_id, we use timestamp to keep context
                _ = self.client.add_message(
                    messages=messages, 
                    user_id=user_id, 
                    conversation_id=timestamp
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    print(f"[Error] Failed to add raw log for {user_id}: {e}")

    def add_memories_for_speaker(self, speaker_user_id, messages, timestamp, desc):
        """
        完全模仿 add.py 的批量处理逻辑，但去掉了 preprocessor。
        直接存入原始的 messages 列表。
        """
        
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            
            # 这里不需要 preprocessor，直接存入原始的 batch
            # 格式已经是 [{"role": "user", "content": "Speaker: Text"}, ...]
            self.add_raw_memory(speaker_user_id, batch_messages, timestamp)

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"


        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages_for_a = []
            messages_for_b = []
            
            # 仿照 add.py 构建第一人称视角的对话列表
            # 这样存储的好处是：Agent 知道 "user" 是自己，"assistant" 是对方
            for chat in chats:
                text_content = f"{chat['speaker']}: {chat['text']}"
                
                if chat["speaker"] == speaker_a:
                    # 对于 Speaker A 来说，自己是 user
                    messages_for_a.append({"role": "user", "content": text_content, "chat_time": timestamp})
                    # 对于 Speaker B 来说，A 是 assistant (对方)
                    messages_for_b.append({"role": "assistant", "content": text_content, "chat_time": timestamp})
                elif chat["speaker"] == speaker_b:
                    # 对于 Speaker A 来说，B 是 assistant (对方)
                    messages_for_a.append({"role": "assistant", "content": text_content, "chat_time": timestamp})
                    # 对于 Speaker B 来说，自己是 user
                    messages_for_b.append({"role": "user", "content": text_content, "chat_time": timestamp})
                    
            # 双线程同时为 A 和 B 添加记忆
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(
                    speaker_a_user_id,
                    messages_for_a,
                    timestamp,
                    f"Adding Raw Logs for {speaker_a}",
                ),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(
                    speaker_b_user_id,
                    messages_for_b,
                    timestamp,
                    f"Adding Raw Logs for {speaker_b}",
                ),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

    def run(self, max_workers=5):
        if not self.data:
            raise ValueError("No data loaded.")
            
        print(f"📦 Starting RAW Conversation Ingestion (to Origin DB)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()
        print("✅ Raw Ingestion Complete.")

if __name__ == "__main__":
    adder = RawConversationAdder()
    adder.run()