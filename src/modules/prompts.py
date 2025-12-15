# Context Engineering: Absolute Time Injection
# 将相对时间（昨天、下周）转化为绝对时间戳，解决 LOCOMO 的时序混淆问题。
FACT_EXTRACTION_PROMPT = """
You are a specialized Memory Encoder. Your task is to extract atomic, time-aware facts from a conversation fragment.

User Name: {user_name}
Conversation Absolute Date: {date}

Instructions:
1. Analyze the conversation between {user_name} and the other speaker.
2. Extract important personal information about {user_name} (events, preferences, plans, relationships, emotions).
3. CRITICAL: Resolve all relative time references (e.g., "yesterday", "next friday") into ABSOLUTE DATES based on the Conversation Date provided above.
4. Ignore greetings and small talk.
5. Format the output as a bulleted list where each line starts with the date in [YYYY-MM-DD] format.

Example Input:
Date: 2022-01-15
User: I went to the gym yesterday and I will fly to Paris in 3 days.

Example Output:
- [2022-01-14] {user_name} went to the gym.
- [2022-01-18] {user_name} plans to fly to Paris.

Current Conversation:
{conversation_text}

Extracted Facts:
"""

# Context Engineering: Query Rewriting
# 解决指代消解（Coreference Resolution）问题。
QUERY_REWRITE_PROMPT = """
You are a Search Optimization Agent.
Original Question: "{question}"
Target Person: {user_name}

The original question might be vague (e.g., "Where did he go?").
Task: Rewrite the question to be self-contained and explicit for a semantic search engine.
1. Replace pronouns (he/she/you) with the Target Person's name: {user_name}.
2. Keep the intent of the question strictly unchanged.
3. Output ONLY the rewritten question string.

Rewritten Query:
"""

# Agent Logic: Temporal Reasoning
# 强制模型根据检索到的时间戳进行推理。
# 3. 最终回答 Prompt (优化版)
# 强制使用 XML 标签包裹答案，方便后续代码提取
ANSWER_PROMPT = """
You are a highly intelligent QA Agent. You have access to the COMPLETE conversation history and specific retrieved memory highlights.

=== PART 1: COMPLETE CONVERSATION HISTORY ===
(This is the ground truth timeline of all events)
{full_history}

=== PART 2: RETRIEVED RAW FRAGMENTS ===
(These are potential matches from the database, for reference)
{origin_memories}

=== PART 3: RETRIEVED FACT SUMMARIES ===
(These are processed facts to help reasoning)
{process_memories}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Base your answer primarily on the "COMPLETE CONVERSATION HISTORY".
2. Use "RETRIEVED FRAGMENTS/FACTS" to quickly locate key details or confirm dates.
3. Answer the question directly and concisely.
4. If the question asks for a date, output the date or simply "the last weekend", etc.
5. Output ONLY the answer string.

Answer:
"""