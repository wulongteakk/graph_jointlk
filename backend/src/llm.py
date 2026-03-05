import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.tongyi import ChatTongyi
import boto3
import google.auth

from src.shared.constants import MODEL_VERSIONS
from src.chunk_utils import get_combined_chunks


def get_llm(model_version: str):
    """Retrieve the specified language model based on the model name."""
    env_key = "LLM_MODEL_CONFIG_" + model_version
    env_value = os.environ.get(env_key)
    logging.info("Model: {}".format(env_key))
    model_name = MODEL_VERSIONS[model_version]
    if "Ollama" in model_version:
        # model_name, base_url = env_value.split(",")
        llm = ChatOpenAI(api_key=os.environ.get('OLLAMA_API_KEY'),
                         base_url=os.environ.get('OLLAMA_API_URL'),
                         model=model_name,
                         # top_p=0.7,
                         temperature=0.98)
    elif "glm" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('ZHIPUAI_API_KEY'),
                         base_url=os.environ.get('ZHIPUAI_API_URL'),
                         model=model_name,
                         # top_p=0.7,
                         temperature=0.98)

    elif "moonshot" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('MOONSHOT_API_KEY'),
                         base_url=os.environ.get('MOONSHOT_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "Baichuan" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('BAICHUAN_API_KEY'),
                         base_url=os.environ.get('BAICHUAN_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "yi-large" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('LINGYIWANWU_API_KEY'),
                         base_url=os.environ.get('LINGYIWANWU_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "deepseek" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'),
                         base_url=os.environ.get('DEEPSEEK_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95)
    elif "qwen" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('QWEN_API_KEY'),
                         base_url=os.environ.get('QWEN_API_URL'),
                         model=model_name,
                         top_p=0.7,
                         temperature=0.95
                         )
    elif "Doubao" in MODEL_VERSIONS[model_version]:
        llm = ChatOpenAI(api_key=os.environ.get('DOUBAO_API_KEY'),
                         base_url=os.environ.get('DOUBAO_API_URL'),
                         model=os.environ.get('ENDPOINT_ID'),
                         # top_p=0.7,
                         # temperature=0.95
                         )
    elif "gemini" in model_version:
        credentials, project_id = google.auth.default()
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatVertexAI(
            model_name=model_name,
            convert_system_message_to_human=True,
            credentials=credentials,
            project=project_id,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    elif "openai" in model_version:
        model_name = MODEL_VERSIONS[model_version]
        llm = ChatOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=model_name,
            temperature=0,
        )

    elif "azure" in model_version:
        model_name, api_endpoint, api_key, api_version = env_value.split(",")
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=api_endpoint,
            azure_deployment=model_name,  # takes precedence over model parameter
            api_version=api_version,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )

    elif "anthropic" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatAnthropic(
            api_key=api_key, model=model_name, temperature=0, timeout=None
        )

    elif "fireworks" in model_version:
        model_name, api_key = env_value.split(",")
        llm = ChatFireworks(api_key=api_key, model=model_name)

    elif "groq" in model_version:
        model_name, base_url, api_key = env_value.split(",")
        llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0)

    elif "bedrock" in model_version:
        model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )

        llm = ChatBedrock(
            client=bedrock_client, model_id=model_name, model_kwargs=dict(temperature=0)
        )

    else:
        model_name = "diffbot"
        llm = DiffbotGraphTransformer(
            diffbot_api_key=os.environ.get("DIFFBOT_API_KEY"),
            extract_types=["entities", "facts"],
        )
    logging.info(f"Model created - Model Version: {model_version}")
    return llm, model_name



def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, use_function=True
):
    # 函数内再次导入，避免某些分支合并遗漏顶层 import inspect 导致 NameError
    import inspect as _inspect

    try:
        from src.graph_transformers.llm import LLMGraphTransformer
    except Exception as e:
        logging.warning(f"Fallback to langchain_experimental LLMGraphTransformer due to import error: {e}")
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    futures = []
    graph_document_list = []
    if not use_function:
        node_properties = False
    else:
        node_properties = [
            "description",
            "layer_code",
            "category_code",
            "confidence",
            "reason",
            "evidence",
        ]
    init_params = set(_inspect.signature(LLMGraphTransformer.__init__).parameters.keys())
    init_kwargs = {
        "llm": llm,
        "allowed_nodes": allowedNodes,
        "allowed_relationships": allowedRelationship,
    }
    if "node_properties" in init_params:
        init_kwargs["node_properties"] = node_properties
    if "use_function_call" in init_params:
        init_kwargs["use_function_call"] = use_function

    llm_transformer = LLMGraphTransformer(**init_kwargs)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list


def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    llm, model_name = get_llm(model)
    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
    graph_document_list = get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship
    )
    return graph_document_list