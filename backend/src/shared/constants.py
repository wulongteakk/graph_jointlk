MODEL_VERSIONS = {
    "openai-gpt-3.5": "gpt-3.5-turbo-16k",
    "gemini-1.0-pro": "gemini-1.0-pro-001",
    "gemini-1.5-pro": "gemini-1.5-pro-preview-0514",
    "openai-gpt-4": "gpt-4-0125-preview",
    "diffbot": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "groq-llama3": "llama3-70b-8192",
    "智谱": "glm-4",
    "百川": "Baichuan4",
    "月之暗面": "moonshot-v1-8k",
    "深度求索": "deepseek-chat",
    "零一万物": "yi-large",
    "通义千问": "qwen-long",
    "豆包": "Doubao-pro-32k",
    "Ollama": "qwen2.5:7b-instruct",
    "openai-gpt-4o": "gpt-4o",
}
OPENAI_MODELS = ["gpt-3.5", "gpt-4o", '智谱', "百川", "月之暗面", "深度求索", "零一万物", "通义千问", "豆包",
                 "openai-gpt-3.5", "Ollama", "openai-gpt-4o"]

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
    ("openai-gpt-3.5", 'azure_ai_gpt_35', "gemini-1.0-pro", "gemini-1.5-pro", "groq-llama3", 'groq_llama3_70b',
     'anthropic_claude_3_5_sonnet', 'fireworks_llama_v3_70b', 'bedrock_claude_3_5_sonnet',): 4,
    ("openai-gpt-4", "diffbot", 'azure_ai_gpt_4o', "openai-gpt-4o"): 28,
    ("ollama_llama3"): 2
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
     [c in chunks | {id: c.chunk.id, evidence_id: coalesce(c.chunk.evidence_id, c.chunk.id), score: c.score}] as chunkdetails
WITH d, avg_score, chunkdetails, "" as text
RETURN text, avg_score AS score,
       {source: COALESCE(CASE WHEN d.url CONTAINS "None" THEN d.fileName ELSE d.url END, d.fileName),
        chunkdetails: chunkdetails} as metadata
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
     [c IN chunks | coalesce(c.chunk.text, "")] AS texts, 
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
# -----------------------------------------------------------------------------
# Stage1 通用施工抽取 Schema（升级版：更贴合 HFCSA + 事故链）
# -----------------------------------------------------------------------------
# 说明：
# - 这些是 LLMGraphTransformer 的 allowed_nodes / allowed_relationships，用于“第一阶段实体关系抽取”。
# - 第二阶段 CTP 事故链会基于第一阶段结果进行“受控候选生成 + 只连边”。
# - 为了兼容旧逻辑，保留 legacy 常量。
# -----------------------------------------------------------------------------

CONSTRUCTION_NODE_LABELS_LEGACY = [
    "人员", "班组", "设备", "物料", "作业", "任务",
    "隐患", "风险", "时间", "地点", "环境因素"
]

CONSTRUCTION_REL_TYPES_LEGACY = [
    "执行", "使用", "产生", "存在", "状态是", "发生在", "关联于"
]

# ✅ 升级版（建议默认使用）
# 你可以继续扩充，比如增加：WorkPermit（作业许可）、Inspection（检查验收）、PPE（个体防护）等
CONSTRUCTION_NODE_LABELS = [
    # 上下文实体（ContextEntity）
    "WorkTask",  # 工序/作业活动
    "WorkObject",  # 构件/构筑物/临设
    "Equipment",  # 设备/机具
    "Location",  # 位置/区域/作业面
    "StandardClause",  # 规范条款/制度条款
    "EvidenceUnit",  # 证据单元（文本片段/记录/图片检测结果等）

    # 风险与屏障（建议用于 C 层表达）
    "HazardSource",  # 危险源/危险能量
    "Barrier",  # 工程/管理/个体防护屏障
    "ResourceCondition",  # 作业条件/资源状态
    "HumanState",  # 人员状态（疲劳/注意力/身体状态）
    "ManagementAction",  # 管理动作（交底/许可/验收/旁站/监护）

    # 行为与事件
    "UnsafeAct",  # 不安全行为/违规/错误
    "AccidentEvent",  # 危险事件/顶事件（E）
    "Loss",  # 后果/损失（L）

    # 保留少量中文通用标签，提升适配性
    "人员", "班组", "隐患", "风险", "时间", "环境因素",
]

# 关系类型（你可以把“因果”留给二阶段；一阶段更偏事实/上下文/约束）
CONSTRUCTION_REL_TYPES = [
    # 上下文绑定
    "hasContext",  # 节点 -> ContextEntity
    "supportedBy",  # 节点 -> EvidenceUnit
    "occursAt",  # 节点 -> Location
    "involves",  # 节点 -> 对象/设备/人员/作业

    # 规范与符合性
    "violates",  # 节点 -> StandardClause
    "conformsTo",  # 节点 -> StandardClause

    # 状态与属性
    "hasState",  # 设备/屏障/环境 -> 状态
    "hasHazard",  # 作业/环境 -> 危险源
    "hasBarrier",  # 危险源/作业 -> 屏障

    # 关联（弱语义，兜底）
    "relatedTo",
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
请对以下描述的建筑安全隐患，按1-5分进行【严重性】打分（		
	1（可忽略）： 无伤害，隐患。	
	2（轻微）： 导致轻微伤害（无需休假）、设备轻微损坏	
	3（中等）： 导致轻伤（需休假）、设备中度损坏	
	4（严重）： 导致重伤、永久性残疾、重大设备损坏	
	5（灾难性）： 导致死亡、重大财产损失、严重环境破坏	

）。
请仅返回一个JSON对象，包含分数和简短理由。
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


# ------------------------------------------------------------------------------
# HFACS 风险分类补全 Prompt (LLM as "Classifier")
# ------------------------------------------------------------------------------
HFACS_CLASSIFICATION_PROMPT_TEMPLATE = """
你是建筑安全风险标注专家。请判断给定实体是否属于安全风险因子/隐患，并在属于时按照 HFACS 体系进行分类和属性补全。

HFACS 一级/二级分类（不要修改这些标签）：
- 不安全行为（A）
  - 非故意不当行为（A1）
  - 例行违规（A2）
- 不安全行为的前提条件 (C)
  - 工作准备不足（C1）
  - 安全设施和防护不足（C2）
  - 调度和管理错误（C3）
  - 安装缺陷（C4）
  - 处于不安全状态或环境中的人员 (C5)
  - 外部环境不适宜（C6）
  - 信息错误或疏漏（C7）
- 不安全监管 (S)
  - 监督和责任方面的缺陷（S1）
  - 现场监管不力（S2）
  - 缺乏风险评估和应急管理（S3）
  - 未能及时纠正问题（S4）
- 组织影响（O）
  - 资源管理（O1）
  - 组织氛围（O2）
  - 组织过程（O3）

请输出 JSON，字段含义如下：
- is_risk_factor: true/false，表示该实体是否属于安全风险因子或隐患。
- risk_name: 规范化后的风险因子名称。
- risk_domain: 与 HFACS 分类匹配的领域描述（例如“不安全行为”“不安全监管”）。
- hfacs_level_1: 选择 A/C/S/O 之一。
- hfacs_level_2: 选择对应的二级标签（如 A1、C3、O2 等）。
- severity_level: 1-5 的整数，估计严重程度（
    1（可忽略）： 无伤害，隐患。
	2（轻微）： 导致轻微伤害（无需休假）、设备轻微损坏
	3（中等）： 导致轻伤（需休假）、设备中度损坏
	4（严重）： 导致重伤、永久性残疾、重大设备损坏
	5（灾难性）： 导致死亡、重大财产损失、严重环境破坏	）。
- reason: 简要说明判断依据。

实体名称："{entity_name}"

如果 is_risk_factor=false，请保持其他字段为空字符串或 null。
"
"""

HFCSA_CONTROLLED_CLASSIFICATION_PROMPT_TEMPLATE = """
你是“HFCSA 受控分类器”。你的任务：对文本中抽取到的每个 RiskFactor（风险因子/人因/管理缺陷/前提条件/不安全行为）进行受控分类。

【硬性约束】
1) 你必须为每个 RiskFactor 输出：
   - layer_code: 只能取 ["O","S","C","A"]
   - category_code: 必须从下面“受控 Category 列表”中选择一个，且必须属于对应 layer_code
   - confidence: 0~1 之间的小数
   - reason: 1 句话说明为什么选这个类目（不要长段解释）
   - evidence: 原文中支持判断的最短证据片段（尽量不超过 25 个字）
2) 严禁输出不在列表中的 category_code。
3) 如果无法判断具体落到哪个细分，仍必须选一个最接近的 category_code，并把 confidence 设为 <= 0.55。
4) 输出必须是严格 JSON，且只输出 JSON，不要输出任何其他文字。

【Layer 定义】
- O：组织/制度层（制度、资源、文化、合规治理、分包体系、人力培训等）
- S：监督/管理层（现场监管、风险评估、作业许可、计划组织、整改闭环等）
- C：作业前提与资源层（准备、屏障防护、设备设施、环境、沟通、人员状态等）
- A：现场行为/操作层（错误/违规/冒险行为等）

【受控 Category 列表（code -> layer -> name -> 简述）】
O1 (O) 安全管理体系与制度：责任制/制度/投入
O2 (O) 分包与层级承包管理：资质审查/转包违法分包/以包代管
O3 (O) 人力资源与培训管理：培训取证/入场教育/班组长能力
O4 (O) 组织安全文化与氛围：重进度轻安全/奖惩失效/违章默许
O5 (O) 资源配置与工期压力：赶工压缩/安管人员不足/投入不足
O6 (O) 标准合规与方案治理：规范理解/专项方案审批变更/内审闭环/验收追溯

S1 (S) 监督与责任缺陷：职责不清/责任制落实弱/追责不严
S2 (S) 现场监管不力：巡查旁站监护缺失/违章不制止
S3 (S) 风险评估与应急不足：风险辨识不足/专项方案不足/应急演练不足
S4 (S) 问题纠正不及时：整改不闭环/复查不到位
S5 (S) 计划组织不当：交叉作业冲突/作业许可失效/临变未评审

C1 (C) 工作准备不足：交底不足/任务不熟/工器具准备不足
C2 (C) 关键屏障/安全防护不足：临边洞口/临电/吊装/支护等屏障缺失或不合规
C3 (C) 制度与现场管理错误：顺序安排/交叉作业/人员配置不当
C4 (C) 设备/设施缺陷：脚手架/吊装设备/临电设施缺陷
C5 (C) 人员状态异常：疲劳/身心异常/技能不足
C6 (C) 外部环境不适宜：风雨雪雷电/照明不足/通道障碍
C7 (C) 信息传递或沟通不畅：信息失真/不及时/未告知关键风险控制点

A1 (A) 非故意不当行为（错误）：操作失误/注意力分散/误判风险
A2 (A) 例行违规：不戴PPE/不正确系挂/拆除挪动防护
A3 (A) 例外违规/冒险行为：明知危险仍作业/违章指挥/擅改工艺拆安全装置

【输出 JSON 结构】
[
  {{
    "risk_factor": "<因子名称/短语>",
    "layer_code": "O|S|C|A",
    "category_code": "O1|...|A3",
    "confidence": 0.00,
    "reason": "<一句话原因>",
    "evidence": "<原文证据片段>"
  }}
]
"""

# =============================================================================
# HFCSA + CTP v2.0（严格逐层事故链）扩展
# =============================================================================

# ------------------------------
# 1) HFCSA 层级与类目元信息（供 UI/检索/二阶段候选生成）
# ------------------------------
HFCSA_LAYER_META = {
    "O": {"label": "O_Node", "name": "组织/制度"},
    "S": {"label": "S_Node", "name": "监督/管理"},
    "C": {"label": "C_Node", "name": "作业前提与资源"},
    "A": {"label": "A_Node", "name": "现场行为/操作"},
    "E": {"label": "E_Node", "name": "危险事件/顶事件"},
    "L": {"label": "L_Node", "name": "后果/损失"},
}

# 你已有的 HFCSA 类目库（O1..A3...）如果在别处维护，也可以只在这里维护一个“最小可用集合”
# 用于 stage2 的候选种子化/解释/校验。若你想更全，可以把你开题报告里的全部条目补全到这里。
HFCSA_CATEGORY_META = {
    # --- O 层 ---
    "O1": {"name": "安全治理与制度体系"},
    "O2": {"name": "承包链与分包治理"},
    "O3": {"name": "人员能力与配置"},
    "O4": {"name": "安全文化与激励约束"},
    "O5": {"name": "计划压力与资源约束"},

    # --- S 层 ---
    "S1": {"name": "监督不足"},
    "S2": {"name": "计划与组织不当"},
    "S3": {"name": "未纠正已知问题"},
    "S4": {"name": "监督违规"},

    # --- C 层 ---
    "C1": {"name": "作业准备不足"},
    "C2": {"name": "暴露与作业条件"},
    "C3": {"name": "工程技术屏障与防护资源不足"},
    "C4": {"name": "设备与临时结构状态不良"},
    "C5": {"name": "环境不利"},
    "C6": {"name": "人员状态与协同不足"},

    # --- A 层 ---
    "A1": {"name": "非故意错误（Errors）"},
    "A2": {"name": "违规行为（Violations）"},
    "A3": {"name": "不安全作业组织动作"},
}

# ------------------------------
# 2) CTP v2.0：严格逐层 Allowed transitions
# ------------------------------
CTP_ALLOWED_TRANSITIONS = [
    ("O", "S"),
    ("S", "C"),
    ("C", "A"),
    ("A", "E"),
    ("E", "L"),
]

# ------------------------------
# 3) E/L 元信息
# ------------------------------
CTP_EVENT_META = {
    "E1": {"name": "坠落事件", "accident_type": "Fall"},
    "E2": {"name": "物体打击事件", "accident_type": "Struck-by"},
    "E3": {"name": "坍塌事件", "accident_type": "Collapse"},
    "E4": {"name": "起重伤害事件", "accident_type": "Lifting"},
    "E5": {"name": "触电/电弧事件", "accident_type": "Electric"},
}
CTP_LOSS_META = {
    "L1": {"name": "未遂"},
    "L2": {"name": "轻伤"},
    "L3": {"name": "重伤"},
    "L4": {"name": "死亡"},
}

CTP_ACCIDENT_TYPE_TO_ECODE = {
    "Fall": "E1",
    "Struck-by": "E2",
    "Collapse": "E3",
    "Lifting": "E4",
    "Electric": "E5",
}
CTP_ECODE_TO_ACCIDENT_TYPE = {v: k for k, v in CTP_ACCIDENT_TYPE_TO_ECODE.items()}

# ------------------------------
# 4) 事故模块库（M1~M5）显式化（C_* / A_*）
# ------------------------------
CTP_MODULE_LIBRARY = {
    "Fall": {
        "module_id": "M1",
        "name": "坠落模块",
        "E": "E1",
        "C": [
            {"code": "C_FALL_1", "name": "高处作业暴露存在（临边/洞口/屋面/脚手架/梯子/吊篮）"},
            {"code": "C_FALL_2", "name": "作业面状态不良（湿滑/踏步不稳/承载不足）"},
            {"code": "C_FALL_3", "name": "集体防护不满足（临边防护/洞口盖板/安全网/平台完整性）"},
            {"code": "C_FALL_4", "name": "个人防护条件不满足（安全带/生命线/挂点/检验不可用）"},
        ],
        "A": [
            {"code": "A_FALL_1", "name": "未系安全带/未正确使用坠落防护"},
            {"code": "A_FALL_2", "name": "翻越护栏/进入未隔离区"},
            {"code": "A_FALL_3", "name": "不当攀爬/不当站位/过度探身"},
            {"code": "A_FALL_4", "name": "擅自拆除或移动临边/洞口防护"},
        ],
    },
    "Struck-by": {
        "module_id": "M2",
        "name": "物体打击模块",
        "E": "E2",
        "C": [
            {"code": "C_SB_1", "name": "高处坠物风险暴露（高处作业/堆放/抛掷）"},
            {"code": "C_SB_2", "name": "隔离与防护不足（警戒区/防护棚/踢脚板/工具系挂条件）"},
            {"code": "C_SB_3", "name": "吊装/车辆/机械运动区暴露（回转半径/盲区）"},
        ],
        "A": [
            {"code": "A_SB_1", "name": "未按规定进入隔离区/站位不当"},
            {"code": "A_SB_2", "name": "抛掷物料/工具未系挂"},
            {"code": "A_SB_3", "name": "指挥/协同错误导致误入危险区"},
        ],
    },
    "Collapse": {
        "module_id": "M3",
        "name": "坍塌模块",
        "E": "E3",
        "C": [
            {"code": "C_COL_1", "name": "临时结构/支护体系状态不良（模板支撑/脚手架/基坑支护）"},
            {"code": "C_COL_2", "name": "超载/堆载控制不足（堆料/设备布置不当）"},
            {"code": "C_COL_3", "name": "监测与验收不到位（沉降位移监测/节点验收）"},
            {"code": "C_COL_4", "name": "地基与排水条件不良"},
        ],
        "A": [
            {"code": "A_COL_1", "name": "违规拆撑/违规拆模/拆改顺序错误"},
            {"code": "A_COL_2", "name": "超载堆放/在禁令条件下作业"},
            {"code": "A_COL_3", "name": "未按方案施工/临时变更做法"},
        ],
    },
    "Lifting": {
        "module_id": "M4",
        "name": "起重伤害模块",
        "E": "E4",
        "C": [
            {"code": "C_LIFT_1", "name": "起重作业暴露存在（起重机/吊具索具/悬吊荷载）"},
            {"code": "C_LIFT_2", "name": "设备与索具状态不良/检验不足"},
            {"code": "C_LIFT_3", "name": "地基承载/支腿/风速等条件不满足"},
            {"code": "C_LIFT_4", "name": "禁入隔离与指挥体系不健全（资源/条件）"},
        ],
        "A": [
            {"code": "A_LIFT_1", "name": "超载/斜拉/违章起吊"},
            {"code": "A_LIFT_2", "name": "指挥信号错误/协同失误"},
            {"code": "A_LIFT_3", "name": "人员进入回转半径/站位不当"},
        ],
    },
    "Electric": {
        "module_id": "M5",
        "name": "触电模块",
        "E": "E5",
        "C": [
            {"code": "C_ELEC_1", "name": "临时用电/带电暴露存在（线路/配电箱/电动工具）"},
            {"code": "C_ELEC_2", "name": "接地/漏保/绝缘等屏障失效或缺失"},
            {"code": "C_ELEC_3", "name": "潮湿/狭小/导电环境不利"},
            {"code": "C_ELEC_4", "name": "停送电管理与验电条件不满足（资源/条件）"},
        ],
        "A": [
            {"code": "A_ELEC_1", "name": "未验电/未挂牌上锁即作业"},
            {"code": "A_ELEC_2", "name": "擅自接电/违规操作电气设备"},
            {"code": "A_ELEC_3", "name": "未保持安全距离/误碰带电体"},
        ],
    },
}

# ------------------------------
# 5) 第二阶段 Prompt（只连边，不造点；严格逐层；必须引用候选实体）
# ------------------------------
CTP_EVENT_CLASSIFIER_PROMPT = """你是施工安全事故类型判别助手。
给你一份事故报告的证据摘要（来自证据库/原文片段），请在以下 5 类中选择最匹配的一类：
- Fall（坠落）
- Struck-by（物体打击）
- Collapse（坍塌）
- Lifting（起重伤害）
- Electric（触电/电弧）

要求：
1) 只输出 JSON（不要 markdown）。
2) accident_type 必须是上面 5 个之一。
3) event_code 必须严格对应：Fall->E1, Struck-by->E2, Collapse->E3, Lifting->E4, Electric->E5
4) confidence 取 0~1 的小数。
5) reason 用一句话说明依据（尽量引用摘要中关键词）。

输入证据摘要（JSON）：
{evidence_summary_json}

输出 JSON schema:
{"accident_type":"Fall","event_code":"E1","confidence":0.0,"reason":"..."}
"""

CTP_EXPERT_PROMPT_TEMPLATE_BASE = """你是 HFCSA+CTP v2.0 事故因果链专家，目标是：
- **只使用给定候选实体（candidate_table）中已有的节点**；
- **只在这些节点之间创建因果边**（nextLevelCauses）；
- **严格逐层：O→S→C→A→E→L**；
- **禁止跳层/逆向/同层/闭环**；
- **每条边必须能用证据摘要支撑，并指明 evidence_chunk_ids**；
- **禁止凭空生成候选表里不存在的实体/关系**。

你将得到两份输入：
1) evidence_summary_json：证据子图摘要（包含 chunk_id 对应的原文片段摘要、关键实体、已有关系概览）
2) candidate_table_json：候选实体表（仅这些节点允许出现在事故链里）

输出要求：
- 只输出 JSON（不要 markdown，不要解释）。
- 输出必须符合 schema：
{
  "case_id": str,
  "selected_event": {"accident_type": str, "event_code": str, "module_id": str},
  "edges": [
     {"from": str, "to": str, "rel": "nextLevelCauses", "evidence_chunk_ids": [str], "justification": str}
  ]
}

强制约束：
- edges 里的 from/to 必须都出现在 candidate_table_json.candidates[].id 中；
- rel 必须是 nextLevelCauses；
- 边必须满足 allowed_transitions（candidate_table_json.allowed_transitions）；
- 事故链必须至少包含一条 A→E 和一条 E→L；
- 若 candidate_table_json.module_nodes 非空，则事故链中必须至少使用 1 个模块 C_* 节点和 1 个模块 A_* 节点；
- evidence_chunk_ids 必须来自 candidate_table_json.available_chunk_ids；
- justification 不超过 40 字，且不得出现“推测/猜测/可能/大概”等字眼。

=====================
evidence_summary_json:
{evidence_summary_json}

=====================
candidate_table_json:
{candidate_table_json}
"""

# ------------------------------
# 6) 事故类型（E）规则投票关键词（用于 LLM+规则融合）
# ------------------------------
# 说明：
# - list 里既支持 "字符串"，也支持 {kw:..., w:...} 带权重写法
# - 这些关键词用于在 Stage1 已抽到的实体 + 原文摘要中做投票，降低纯 LLM 误判
CTP_EVENT_RULE_KEYWORDS = {
    "Fall": [
        {"kw": "坠落", "w": 2.5},
        {"kw": "高处", "w": 1.8},
        "临边",
        "洞口",
        "脚手架",
        "梯子",
        "吊篮",
        "安全带",
        "生命线",
        "挂点",
        "安全网",
        "坠亡",
    ],
    "Struck-by": [
        {"kw": "物体打击", "w": 2.5},
        {"kw": "坠物", "w": 2.0},
        "落物",
        "砸伤",
        "飞来物",
        "抛掷",
        "警戒区",
        "防护棚",
        "踢脚板",
        "回转半径",
        "盲区",
    ],
    "Collapse": [
        {"kw": "坍塌", "w": 2.5},
        {"kw": "倒塌", "w": 2.0},
        "塌方",
        "模板",
        "支撑",
        "支护",
        "基坑",
        "沉降",
        "位移",
        "失稳",
        "脚手架坍塌",
    ],
    "Lifting": [
        {"kw": "吊装", "w": 2.5},
        {"kw": "起重", "w": 2.2},
        "塔吊",
        "吊车",
        "起重机",
        "吊具",
        "索具",
        "钢丝绳",
        "超载",
        "斜拉",
        "指挥信号",
        "回转",
    ],
    "Electric": [
        {"kw": "触电", "w": 2.5},
        {"kw": "电弧", "w": 2.2},
        "带电",
        "配电箱",
        "漏保",
        "漏电保护器",
        "接地",
        "绝缘",
        "电缆",
        "短路",
        "验电",
        "挂牌",
        "上锁",
    ],
}

# ------------------------------
# 7) 严重度（L）规则关键词
# ------------------------------
CTP_SEVERITY_RULE_KEYWORDS = {
    "L4": ["死亡", "身亡", "当场死亡", "致死", "死亡1人", "死亡2人", "死亡3人", "死亡4人"],
    "L3": ["重伤", "危重", "严重受伤", "多处骨折", "昏迷", "休克"],
    "L2": ["轻伤", "受伤", "擦伤", "扭伤", "割伤", "骨折"],
    "L1": ["未遂", "险情", "事故未遂", "险些", "未造成人员伤亡", "无人员伤亡"],
    "NO_CASUALTY": ["无人员伤亡", "未造成人员伤亡", "未造成伤亡", "未发生伤亡"],
}

# ------------------------------
# 8) 严重度（L）LLM 判别 prompt（用于规则不确定时兜底）
# ------------------------------
CTP_SEVERITY_CLASSIFIER_PROMPT = """你是施工安全事故严重度（后果）判别助手。
给你一段事故报告证据摘要，请在以下等级中选择最匹配的一个：
- L1 未遂
- L2 轻伤
- L3 重伤
- L4 死亡

要求：
1) 只输出 JSON（不要 markdown）。
2) severity_code 必须是 L1/L2/L3/L4 之一。
3) confidence 取 0~1 的小数。
4) reason 用一句话说明依据（尽量引用摘要中的关键词）。

输入证据摘要（JSON）：
{evidence_summary_json}

输出 JSON schema:
{"severity_code":"L1","confidence":0.0,"reason":"..."}
"""
