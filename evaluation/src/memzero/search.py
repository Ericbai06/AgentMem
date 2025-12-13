import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

# from mem0 import MemoryClient
from memos.api.client import MemOSClient

load_dotenv()


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        self.client = MemOSClient(
            api_key=os.getenv("MEMOS_API_KEY")
        )
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                # MemOS requires conversation_id. Using a generic one for search context.
                conversation_id = "search_context"
                memories = self.client.search_memory(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id
                )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        
        semantic_memories = []

        # Helper function to process memory details
        def process_memory_details(memory_list):
            processed = []
            if not memory_list:
                return processed
            for detail in memory_list:
                # Handle object-like detail (from SDK response)
                if hasattr(detail, 'memory_value'):
                    processed.append({
                        "memory": detail.memory_value,
                        "timestamp": detail.conversation_id, # Using conversation_id as timestamp based on your add.py logic
                        "score": round(detail.relativity, 2) if hasattr(detail, 'relativity') and detail.relativity is not None else 0.0
                    })
                # Handle dict-like detail (if SDK returns dicts)
                elif isinstance(detail, dict):
                    processed.append({
                        "memory": detail.get('memory_value', ''),
                        "timestamp": detail.get('conversation_id', 'Unknown Time'),
                        "score": round(detail.get('relativity', 0.0), 2)
                    })
            return processed

        # 1. Handle SDK object response (SearchMemoryResponse or similar)
        if hasattr(memories, 'data'):
            data_obj = memories.data
            if hasattr(data_obj, 'memory_detail_list'):
                semantic_memories.extend(process_memory_details(data_obj.memory_detail_list))
        
        # 2. Handle tuple response (e.g. ('data', SearchMemoryData(...)))
        elif isinstance(memories, tuple):
            for item in memories:
                if hasattr(item, 'memory_detail_list'):
                    semantic_memories.extend(process_memory_details(item.memory_detail_list))
        
        # 3. Handle dictionary response (raw JSON)
        elif isinstance(memories, dict):
            data = memories.get('data', {})
            if isinstance(data, dict):
                memory_list = data.get('memory_detail_list', [])
                semantic_memories.extend(process_memory_details(memory_list))
            # Fallback: maybe 'results' key from previous logic
            elif 'results' in memories:
                 # ... existing fallback logic if needed, but let's stick to the new spec ...
                 pass

        # 4. Handle list response (if it returns a list of results directly)
        elif isinstance(memories, list):
             # Check if items in list are the data objects
             for item in memories:
                 if hasattr(item, 'memory_detail_list'):
                     semantic_memories.extend(process_memory_details(item.memory_detail_list))
                 elif isinstance(item, tuple) and len(item) > 1 and hasattr(item[1], 'memory_detail_list'):
                     semantic_memories.extend(process_memory_details(item[1].memory_detail_list))

        graph_memories = None

        graph_memories = None
        
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Slice data to only process the first 2 items
        data = data[:2]

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                # Save results after each question is processed
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
