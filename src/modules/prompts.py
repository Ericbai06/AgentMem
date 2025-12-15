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
0. **Output format (STRICT)**:
   - Output ONE single line.
   - Output ONLY the answer text (no "Answer:", no quotes, no bullets, no markdown, no extra explanation).
   - Avoid parentheses/brackets. Do not include multiple lines.

1. **Date format is CRITICAL**: 
   - ALWAYS calculate the specific calendar date (e.g., "15 June, 2023") based on the conversation history. 
   - DO NOT use relative terms like "yesterday", "last week", "a few days ago", or "this month" unless the specific date is impossible to determine.
   - If the specific day is unknown, output the Month and Year (e.g., "June 2023").

2. **Extraction over Abstraction**:
   - Use the exact keywords or phrases from the text rather than summarizing them into high-level concepts (e.g., if text says "made posters", say "made posters", NOT "marketing campaign").
   - If the text uses a specific short phrase (e.g., "magical"), use that exact word. Do not add extra adjectives.

3. **Completeness for Lists**:
   - If the question asks for multiple items (e.g., "What events...", "How did she..."), list ALL key items found in the text, separated by commas.
   - For lists, you may exceed the word limit to ensure all items are included.

4. **Conciseness**:
   - For simple fact questions, keep the answer under 10 words.
   - Do NOT use full sentences. Output only the entity, action, or date.

=== EXAMPLES (Follow this style STRICTLY) ===
Q: When did he go to Paris?
Bad Answer: He went last week (January 5th).
Bad Answer: Last Friday.
Good Answer: 5 January, 2022

Q: How did she promote the store?
Bad Answer: She used marketing strategies and influencers.
Good Answer: worked with an artist, made limited-edition sweatshirts, developed a video

Q: What is his feeling?
Bad Answer: He feels excited, happy and full of joy.
Good Answer: excited

Q: What items did he buy?
Bad Answer: groceries
Good Answer: milk, eggs, bread

=== YOUR ANSWER ===
Output:
"""
