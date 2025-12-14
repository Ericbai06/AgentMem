import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from memos.api.client import MemOSClient
from .config import Config
from .prompts import QUERY_REWRITE_PROMPT, ANSWER_PROMPT

class LocomoAgent:
    def __init__(self):
        # 双客户端初始化
        self.client_origin = MemOSClient(api_key=Config.MEMOS_ORIGIN_API_KEY)
        self.client_process = MemOSClient(api_key=Config.MEMOS_PROCESS_API_KEY)
        
        self.llm_client = OpenAI(api_key=Config.OPENAI_API_KEY, base_url=Config.OPENAI_BASE_URL)
        self.results = defaultdict(list)
        self.output_file = os.path.join(Config.RESULTS_DIR, "final_results.json")

    def rewrite_query(self, question, user_name):
        try:
            prompt = QUERY_REWRITE_PROMPT.format(question=question, user_name=user_name)
            response = self.llm_client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except:
            return question

    def _parse_search_response(self, res, source_type):
        """通用解析函数"""
        memories = []
        data_list = []
        # 兼容 SDK 不同版本的返回结构
        if hasattr(res, 'data') and hasattr(res.data, 'memory_detail_list'):
            data_list = res.data.memory_detail_list
        elif isinstance(res, dict):
            data_list = res.get('data', {}).get('memory_detail_list', [])
        elif isinstance(res, list): # 某些情况直接返回 list
            data_list = res
            
        for item in data_list:
            # 健壮性获取
            content = getattr(item, 'memory_value', None) or (item.get('memory_value') if isinstance(item, dict) else "")
            ts = getattr(item, 'conversation_id', None) or (item.get('conversation_id') if isinstance(item, dict) else "Unknown Date")
            
            if content:
                memories.append({
                    "timestamp": ts, 
                    "content": content, 
                    "source": source_type
                })
        return memories

    def search_memories(self, user_id, query):
        try:
            # 并行检索
            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(self.client_origin.search_memory, query, user_id, "search")
                f2 = executor.submit(self.client_process.search_memory, query, user_id, "search")
                
                res_origin = f1.result()
                res_process = f2.result()
            
            mems_origin = self._parse_search_response(res_origin, "RAW")
            mems_process = self._parse_search_response(res_process, "FACT")
            
            all_memories = mems_origin + mems_process
            # 按时间戳简单排序
            all_memories.sort(key=lambda x: str(x.get('timestamp', '')))
            
            return all_memories
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def answer_question(self, user_id, user_name, question):
        # 1. 重写问题
        rewritten_q = self.rewrite_query(question, user_name)
        
        # 2. 双路检索
        memories = self.search_memories(user_id, rewritten_q)
        
        # 3. 构建上下文
        context_lines = []
        for m in memories:
            # 格式：[2022-01-01] Content
            context_lines.append(f"[{m['timestamp']}] {m['content']}")
        
        context_str = "\n".join(context_lines)
        
        # 4. 生成答案 (Direct Answer)
        prompt = ANSWER_PROMPT.format(question=question, context=context_str)
        
        start_t = time.time()
        response = self.llm_client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0, # 保持0温确保简洁
            max_tokens=50    # 限制 token 输出长度，强制模型短答
        )
        duration = time.time() - start_t
        
        # 直接拿内容，不做任何处理
        final_answer = response.choices[0].message.content.strip()
        
        return final_answer, memories, duration

    def process_one_qa(self, qa_item, speaker_a_id, speaker_b_id, spk_a_name, spk_b_name):
        question = qa_item["question"]
        # 直接获取答案
        ans_a, mems_a, _ = self.answer_question(speaker_a_id, spk_a_name, question)
        
        return {
            "question": question,
            "answer": qa_item.get("answer", ""),
            "category": qa_item.get("category", ""),
            "response": ans_a, # 直接存入模型输出
            "evidence": [],
            "speaker_1_memories": mems_a,
            "response_time": 0
        }

    def run_eval(self):
        print(f"🚀 Starting Fast-Track Evaluation (Direct Answer)...")
        with open(Config.DATA_PATH, "r") as f:
            raw_data = json.load(f)
            
        for idx, item in tqdm(enumerate(raw_data), total=len(raw_data)):
            spk_a = item["conversation"]["speaker_a"]
            spk_b = item["conversation"]["speaker_b"]
            uid_a = f"{spk_a}_{idx}"
            uid_b = f"{spk_b}_{idx}"
            qa_list = item["qa"]
            
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS_SEARCH) as executor:
                futures = []
                for qa in qa_list:
                    futures.append(executor.submit(self.process_one_qa, qa, uid_a, uid_b, spk_a, spk_b))
                
                for f in futures:
                    res = f.result()
                    self.results[idx].append(res)
            
            with open(self.output_file, "w") as f:
                json.dump(self.results, f, indent=4)
        print(f"✅ Evaluation Complete. Results saved to {self.output_file}")