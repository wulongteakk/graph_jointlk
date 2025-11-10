MODEL_VERSIONS = {
        "openai-gpt-3.5": "gpt-3.5-turbo-16k",
        "gemini-1.0-pro": "gemini-1.0-pro-001",
        "gemini-1.5-pro": "gemini-1.5-pro-preview-0514",
        "openai-gpt-4": "gpt-4-0125-preview",
        "diffbot" : "gpt-4o",
        "gpt-4o":"gpt-4o",
        "groq-llama3" : "llama3-70b-8192",
        "智谱": "glm-4",
        "百川": "Baichuan4",
        "月之暗面": "moonshot-v1-8k",
        "深度求索": "deepseek-chat",
        "零一万物": "yi-large",
        "通义千问": "qwen-long",
        "豆包": "Doubao-pro-32k",
        "Ollama": "qwen2:1.5b",
        "openai-gpt-4o":"gpt-4o",
         }
OPENAI_MODELS = ["gpt-3.5", "gpt-4o",'智谱',"百川","月之暗面","深度求索","零一万物","通义千问","豆包","openai-gpt-3.5", "Ollama", "openai-gpt-4o"]

GEMINI_MODELS = ["gemini-1.0-pro", "gemini-1.5-pro"]
GROQ_MODELS = ["groq-llama3"]
BUCKET_UPLOAD = 'llm-graph-builder-upload'
BUCKET_FAILED_FILE = 'llm-graph-builder-failed'
PROJECT_ID = 'llm-experiments-387609' 


## CHAT SETUP
CHAT_MAX_TOKENS = 1000
CHAT_SEARCH_KWARG_K = 3
CHAT_SEARCH_KWARG_SCORE_THRESHOLD = 0.7
CHAT_DOC_SPLIT_SIZE = 3000
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD = 0.10
CHAT_TOKEN_CUT_OFF = {
     ("openai-gpt-3.5",'azure_ai_gpt_35',"gemini-1.0-pro","gemini-1.5-pro","groq-llama3",'groq_llama3_70b','anthropic_claude_3_5_sonnet','fireworks_llama_v3_70b','bedrock_claude_3_5_sonnet', ) : 4, 
     ("openai-gpt-4","diffbot" ,'azure_ai_gpt_4o',"openai-gpt-4o") : 28,
     ("ollama_llama3") : 2  
} 

EXTRACTION_PROMPT_TEMPLATE = """
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts.
- **Relationships** represent connections between entities.
Extract all entities and relationships from the text.
"""

### CHAT TEMPLATES 
CHAT_SYSTEM_TEMPLATE = """
You are an AI-powered question-answering agent. Your task is to provide accurate and comprehensive responses to user queries based on the given context, chat history, and available resources.

### Response Guidelines:
1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context provided below.
3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
5. **Avoid Hallucination**: Only provide information based on the context provided. Do not invent information.
6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
9. **Fallback Options**: If the required information is not available in the provided context, provide a polite and helpful response. Example: "I don't have that information right now." or "I'm sorry, but I don't have that information. Is there something else I can help with?"
10. **Context Availability**: If the context is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.


**IMPORTANT** : DO NOT ANSWER FROM YOUR KNOWLEDGE BASE USE THE BELOW CONTEXT

### Context:
<context>
{context}
</context>

### Example Responses:
User: Hi 
AI Response: 'Hello there! How can I assist you today?'

User: "What is Langchain?"
AI Response: "Langchain is a framework that enables the development of applications powered by large language models, such as chatbots. It simplifies the integration of language models into various applications by providing useful tools and components."

User: "Can you explain how to use memory management in Langchain?"
AI Response: "Langchain's memory management involves utilizing built-in mechanisms to manage conversational context effectively. It ensures that the conversation remains coherent and relevant by maintaining the history of interactions and using it to inform responses."

User: "I need help with PyCaret's classification model."
AI Response: "PyCaret simplifies the process of building and deploying machine learning models. For classification tasks, you can use PyCaret's setup function to prepare your data. After setup, you can compare multiple models to find the best one, and then fine-tune it for better performance."

User: "What can you tell me about the latest realtime trends in AI?"
AI Response: "I don't have that information right now. Is there something else I can help with?"

Note: This system does not generate answers based solely on internal knowledge. It answers from the information provided in the user's current and previous inputs, and from the context.
"""


QUESTION_TRANSFORM_TEMPLATE = "Given the below conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else." 


## CHAT QUERIES 
VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, collect(distinct {chunk: chunk, score: score}) as chunks, avg(score) as avg_score
WITH d, avg_score, 
     [c in chunks | c.chunk.text] as texts, 
     [c in chunks | {id: c.chunk.id, score: c.score}] as chunkdetails
WITH d, avg_score, chunkdetails,
     apoc.text.join(texts, "\n----\n") as text
RETURN text, avg_score AS score, 
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} as metadata
""" 

# VECTOR_GRAPH_SEARCH_QUERY="""
# WITH node as chunk, score
# MATCH (chunk)-[:PART_OF]->(d:Document)
# CALL { WITH chunk
# MATCH (chunk)-[:HAS_ENTITY]->(e)
# MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document)
# UNWIND rels as r
# RETURN collect(distinct r) as rels
# }
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts, 
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,  
# 	[r in rels | coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ startNode(r).id + " "+ type(r) + " " + coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + endNode(r).id] as entities
# WITH d, avg_score,chunkdetails,
# apoc.text.join(texts,"\n----\n") +
# apoc.text.join(entities,"\n")
# as text
# RETURN text, avg_score AS score, {source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """  


# VECTOR_GRAPH_SEARCH_QUERY = """
# WITH node as chunk, score
# // find the document of the chunk
# MATCH (chunk)-[:PART_OF]->(d:Document)
# // fetch entities
# CALL { WITH chunk
# // entities connected to the chunk
# // todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
# MATCH (chunk)-[:HAS_ENTITY]->(e)

# // depending on match to query embedding either 1 or 2 step expansion
# WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
# THEN 
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,1}(:!Chunk&!Document) RETURN path }
# ELSE 
# collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document) RETURN path } 
# END as paths

# RETURN collect{ unwind paths as p unwind relationships(p) as r return distinct r} as rels,
# collect{ unwind paths as p unwind nodes(p) as n return distinct n} as nodes
# }
# // aggregate chunk-details and de-duplicate nodes and relationships
# WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels,

# // TODO sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
# apoc.coll.toSet(apoc.coll.flatten(collect(
#                 [r in rels |[startNode(r),endNode(r)]]),true)) as nodes

# // generate metadata and text components for chunks, nodes and relationships
# WITH d, avg_score,
#      [c IN chunks | c.chunk.text] AS texts, 
#      [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,  
#   apoc.coll.sort([n in nodes | 

# coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+ 
# n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
# 	apoc.coll.sort([r in rels 
#     // optional filter if we limit the node-set
#     // WHERE startNode(r) in nodes AND endNode(r) in nodes 
#   | 
# coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ 
# startNode(r).id +
# " " + type(r) + " " + 
# coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + 
# endNode(r).id
# ]) as relTexts

# // combine texts into response-text
# WITH d, avg_score,chunkdetails,
# "Text Content:\n" +
# apoc.text.join(texts,"\n----\n") +
# "\n----\nEntities:\n"+
# apoc.text.join(nodeTexts,"\n") +
# "\n----\nRelationships:\n"+
# apoc.text.join(relTexts,"\n")

# as text
# RETURN text, avg_score as score, {length:size(text), source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails} AS metadata
# """

VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 25

VECTOR_GRAPH_SEARCH_QUERY = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)

// aggregate chunk-details
WITH d, collect(DISTINCT {{chunk: chunk, score: score}}) AS chunks, avg(score) as avg_score
// fetch entities
CALL {{ WITH chunks
UNWIND chunks as chunkScore
WITH chunkScore.chunk as chunk
// entities connected to the chunk
// todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
// todo sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
WITH e, count(*) as numChunks 
ORDER BY numChunks DESC LIMIT {no_of_entites}
// depending on match to query embedding either 1 or 2 step expansion
WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
THEN 
collect {{ OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document) RETURN path }}
ELSE 
collect {{ OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document) RETURN path }} 
END as paths, e
WITH apoc.coll.toSet(apoc.coll.flatten(collect(distinct paths))) as paths, collect(distinct e) as entities
// de-duplicate nodes and relationships across chunks
RETURN collect{{ unwind paths as p unwind relationships(p) as r return distinct r}} as rels,
collect{{ unwind paths as p unwind nodes(p) as n return distinct n}} as nodes, entities
}}

// generate metadata and text components for chunks, nodes and relationships
WITH d, avg_score,
     [c IN chunks | c.chunk.text] AS texts, 
     [c IN chunks | {{id: c.chunk.id, score: c.score}}] AS chunkdetails, 
  apoc.coll.sort([n in nodes | 

coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+ 
n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
	apoc.coll.sort([r in rels 
    // optional filter if we limit the node-set
    // WHERE startNode(r) in nodes AND endNode(r) in nodes 
  | 
coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ 
startNode(r).id +
" " + type(r) + " " + 
coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + endNode(r).id
]) as relTexts
, entities
// combine texts into response-text

WITH d, avg_score,chunkdetails,
"Text Content:\\n" +
apoc.text.join(texts,"\\n----\\n") +
"\\n----\\nEntities:\\n"+
apoc.text.join(nodeTexts,"\\n") +
"\\n----\\nRelationships:\\n" +
apoc.text.join(relTexts,"\\n")

as text,entities

RETURN text, avg_score as score, {{length:size(text), source: COALESCE( CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName), chunkdetails: chunkdetails}} AS metadata
"""

# ==============================================================================
# START: 建筑安全风险量化 (Construction Safety Risk Quantification)
# ==============================================================================

# 阶段一：Schema 定义 (Node and Relationship Types)
CONSTRUCTION_NODE_LABELS = [
    "人员", "班组", "设备", "物料", "作业", "任务",
    "隐患", "风险", "时间", "地点", "环境因素"
]

CONSTRUCTION_REL_TYPES = [
    "执行", "使用", "产生", "存在", "状态是", "发生在", "关联于"
]

# ------------------------------------------------------------------------------
# 阶段二：知识提取 Prompt (LLM as "Information Extractor")
# ------------------------------------------------------------------------------
# 这个 Prompt 将用于 LLMGraphTransformer
CONSTRUCTION_EXPERT_PROMPT_TEMPLATE = """
你是一个建筑安全专家和信息提取算法。
请严格按照我定义的 Schema，从以下报告文本中提取所有相关的实体和关系。
**必须**以我要求的 JSON 格式输出，不要包含任何解释或额外的文本。

Schema (本体) 定义:
节点 (Entities): {node_labels}
关系 (Relations): {rel_types}

报告原文:
---
{input}
---

输出格式 (严格遵守):
{{
  "entities": [
    {{"id": "实体ID或名称", "type": "节点类型", "name": "实体名称", "properties": {{"status": "状态(可选)"}} }}
  ],
  "relations": [
    {{"from_id": "源实体ID", "type": "关系类型", "to_id": "目标实体ID"}}
  ]
}}

示例输入:
'10月25日上午，安全员李四巡查A区3号楼，发现电焊组张三在未佩戴防护面罩的情况下进行动火作业（隐患编号A-102）。已当场责令停止。'

示例输出:
{{
  "entities": [
    {{"id": "10月25日上午", "type": "时间", "name": "10月25日上午", "properties": {{}} }},
    {{"id": "安全员-李四", "type": "人员", "name": "安全员-李四", "properties": {{}} }},
    {{"id": "A区-3号楼", "type": "地点", "name": "A区-3号楼", "properties": {{}} }},
    {{"id": "电焊组-张三", "type": "人员", "name": "电焊组-张三", "properties": {{}} }},
    {{"id": "未佩戴防护面罩", "type": "隐患", "name": "未佩戴防护面罩", "properties": {{}} }},
    {{"id": "动火作业", "type": "作业", "name": "动火作业", "properties": {{}} }},
    {{"id": "A-102", "type": "隐患", "name": "未佩戴防护面罩", "properties": {{"status": "已整改"}} }}
  ],
  "relations": [
    {{"from_id": "安全员-李四", "type": "执行", "to_id": "巡查"}},
    {{"from_id": "电焊组-张三", "type": "执行", "to_id": "动火作业"}},
    {{"from_id": "电焊组-张三", "type": "存在", "to_id": "未佩戴防护面罩"}},
    {{"from_id": "未佩戴防护面罩", "type": "关联于", "to_id": "动火作业"}},
    {{"from_id": "动火作业", "type": "发生在", "to_id": "A区-3号楼"}},
    {{"from_id": "A-102", "type": "关联于", "to_id": "未佩戴防护面罩"}}
  ]
}}
"""

# ------------------------------------------------------------------------------
# 阶段三：隐患打分 Prompt (LLM as "Analyst")
# ------------------------------------------------------------------------------
HAZARD_SCORING_PROMPT_TEMPLATE = """
你是一个资深的建筑安全评估专家。
请对以下描述的建筑安全隐患，按1-10分进行【严重性】打分（1分为轻微，10分为可能导致致命事故）。
请仅返回一个JSON对象，包含分数和简短理由。

隐患描述: "{hazard_description}"

输出格式:
{{"score": <分数>, "reason": "<简短理由>"}}
"""

# ------------------------------------------------------------------------------
# 阶段四：风险概率量化 Prompt (LLM as "Decision-maker")
# ------------------------------------------------------------------------------
RISK_ASSESSMENT_PROMPT_TEMPLATE = """
你是一个顶级的建筑安全事故概率量化模型。
请基于以下从知识图谱中检索到的、与“{query}”相关的实时风险因子，分析它们之间的【耦合效应】，并给出一个最终的【量化风险分数】（0-100分）和【事故概率等级】（极低/低/中/高/极高）。

--- 实时风险因子 ---
{context}
---

请严格按以下 JSON 格式输出你的分析和评估结果：

{{
  "risk_coupling_analysis": "（在这里分析各因子如何相互作用，特别是大雨、基坑异常等如何放大风险，以及缓解措施如何降低风险）",
  "quantitative_risk_score": <一个 0-100 的整数分数>,
  "accident_probability_level": "（极低/低/中/高/极高）"
}}
"""

# ==============================================================================
# END: 建筑安全风险量化
# ==============================================================================