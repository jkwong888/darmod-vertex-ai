import os

from google.cloud import secretmanager
import google.auth

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("GOOGLE_CLOUD_REGION")

EMBED_MODEL_TYPE = os.environ.get("EMBED_MODEL_TYPE", "vertex")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "text-embedding-005")
LLM_MODEL_TYPE = os.environ.get("LLM_MODEL_TYPE", "vertex")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.0-flash-001")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")


# Set up authentication
def get_auth_token():
    creds, project = google.auth.default()

    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    return creds.token


# Create a client
def get_weaviate_api_key() -> str:
    client = secretmanager.SecretManagerServiceClient()

    # Access the secret
    name = f"projects/{PROJECT_ID}/secrets/weaviate_key/versions/latest"
    response = client.access_secret_version(request={"name": name})

    # Extract and print the secret value
    weaviate_key = response.payload.data.decode("UTF-8")
    return weaviate_key

rag_cfg = {
    "project_id": PROJECT_ID,
    "region": REGION,

    # Node parser config
    "chunk_size": 256,
    "chunk_overlap": 0,

    # Embedding model config
    # "embed_model_type": "hf",
    # "embed_model_name": "BAAI/bge-base-en-v1.5",
    # "embed_model_type": "vertex-endpoint",
    # "embed_model_name": "BAAI/bge-base-en-v1.5",
    # "embed_model_endpoint_id": "83814671873736704", # endpoint id
    # "embed_model_use_dedicated_endpoint": True,
    # "embed_model_dedicated_dns": "83814671873736704.us-central1-205512073711.prediction.vertexai.goog",
    "embed_model_type": EMBED_MODEL_TYPE,
    "embed_model_name": EMBED_MODEL_NAME,

    # LLM config
    # "llm_type": "kscope",
    # "llm_name": "Meta-Llama-3.1-8B-Instruct",
    # "llm_type": "vertex-endpoint",
    # "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    # "llm_endpoint_id": "133354267774812160",
    # "llm_use_dedicated_endpoint": True,
    # "llm_dedicated_dns": "133354267774812160.us-central1-205512073711.prediction.vertexai.goog",
    "llm_type": LLM_MODEL_TYPE,
    "llm_name": LLM_MODEL_NAME,
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 50,
    "do_sample": False,

    # Vector DB config
    "vector_db_type": "weaviate", # "weaviate"
    #"vector_db_type": "vertex",
    "vector_db_name": "Pubmed_QA",
    # MODIFY THIS
    "weaviate_url": WEAVIATE_URL,

    # Retriever and query config
    "retriever_type": "vector_index", # "vector_index"
    "retriever_similarity_top_k": 5,
    "query_mode": "default", # "default", "hybrid" - jkwng: changed to default
    "hybrid_search_alpha": 0.0, # float from 0.0 (sparse search - bm25) to 1.0 (vector search)
    "response_mode": "compact",
    "use_reranker": False,
    "rerank_top_k": 3,

    # Evaluation config
    # "eval_llm_type": "kscope",
    # "eval_llm_name": "Meta-Llama-3.1-8B-Instruct",
    "eval_llm_type": "vertex",
    "eval_llm_name": "gemini-2.0-flash-001"
}