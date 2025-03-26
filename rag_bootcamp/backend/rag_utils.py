from google.genai.types import HarmCategory, HarmBlockThreshold

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.google_genai import GoogleGenAI

from llama_index.core import (
    PromptTemplate,
    get_response_synthesizer,
)

from llama_index.embeddings.vertex_endpoint import VertexEndpointEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

from config import get_auth_token

# from .rag_utils import get_embed_model_dim
def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)

def set_query_engine_args(rag_cfg):
    query_engine_args = {
        "similarity_top_k": rag_cfg['retriever_similarity_top_k'],
        "response_mode": rag_cfg['response_mode'],
        "use_reranker": False,
    }

    # jkwng: add that retriever type vector_index could be "vertex" too
    # jkwng: note we don't actually use hybrid search for vertex ai vector search
    if (rag_cfg["retriever_type"] == "vector_index") and (rag_cfg["vector_db_type"] == "weaviate"):
        query_engine_args.update({
            "query_mode": rag_cfg["query_mode"],
            "hybrid_search_alpha": rag_cfg["hybrid_search_alpha"]
        })
    elif (rag_cfg["retriever_type"] == "vector_index") and (rag_cfg["vector_db_type"] == "vertex"):
        query_engine_args.update({
            # jkwng: only default mode works with VVS
            "query_mode": "default",
            "hybrid_search_alpha": 0.0,
        })

    if rag_cfg["use_reranker"]:
        query_engine_args.update({"use_reranker": True, "rerank_top_k": rag_cfg["rerank_top_k"]})

    return query_engine_args


class RAGLLM:
    """
    LlamaIndex supports OpenAI, Cohere, AI21 and HuggingFace LLMs
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    """

    def __init__(self, llm_type, llm_name, api_base=None, api_key=None):
        self.llm_type = llm_type
        self.llm_name = llm_name

        self._api_base = api_base
        self._api_key = api_key

        self.local_model_path = "/model-weights"

    def load_model(self, **kwargs):
        print(f"Configuring {self.llm_type} LLM model ...")
        gen_arg_keys = ["temperature", "top_p", "top_k", "do_sample"]
        gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_arg_keys}
        if self.llm_type == "local":
            # Using local HuggingFace LLM stored at /model-weights
            # llm = HuggingFaceLLM(
            #     tokenizer_name=f"{self.local_model_path}/{self.llm_name}",
            #     model_name=f"{self.local_model_path}/{self.llm_name}",
            #     device_map="auto",
            #     context_window=4096,
            #     max_new_tokens=kwargs["max_new_tokens"],
            #     generate_kwargs=gen_kwargs,
            #     # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
            # )
            pass
        # jkwng: add vertex support
        elif self.llm_type in ["vertex"]:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            llm = GoogleGenAI(
                model=self.llm_name,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_new_tokens"],
                safety_settings=safety_settings,
                vertexai_config={
                  "project": kwargs["project_id"],
                  "location": kwargs["region"],
                },
            )
        elif self.llm_type in ["vertex-endpoint"]:
            ENDPOINT_RESOURCE_NAME = "projects/{}/locations/{}/endpoints/{}".format(
                kwargs["project_id"], kwargs["region"], kwargs["llm_endpoint_id"] # llm_name is the endpoint id
            )
            BASE_URL = (
              f"https://{kwargs['region']}-aiplatform.googleapis.com/v1beta1/{ENDPOINT_RESOURCE_NAME}"
            )
            try:
                if kwargs["llm_use_dedicated_endpoint"]:
                    BASE_URL = f"https://{kwargs['llm_dedicated_dns']}/v1/{ENDPOINT_RESOURCE_NAME}"
            except NameError:
                pass
            llm = OpenAILike(
                model=self.llm_name,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_new_tokens"],
                api_base=BASE_URL,
                api_key=get_auth_token(),
                is_chat_model=True,
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
            )
        elif self.llm_type in ["openai", "kscope"]:
            llm = OpenAILike(
                model=self.llm_name,
                api_base=self._api_base,
                api_key=self._api_key,
                is_chat_model=True,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_new_tokens"],
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
            )
        return llm


class RAGEmbedding:
    """
    LlamaIndex supports embedding models from OpenAI, Cohere, HuggingFace, etc.
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    We can also build out custom embedding model:
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#custom-embedding-model
    """

    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

    def load_model(self, **kwargs):
        print(f"Loading {self.model_type} embedding model ...")
        if self.model_type == "hf":
            # # Using bge base HuggingFace embeddings, can choose others based on leaderboard:
            # # https://huggingface.co/spaces/mteb/leaderboard
            # model = HuggingFaceEmbedding(
            #     model_name=self.model_name,
            #     device="cuda",
            #     trust_remote_code=True,
            # )  # max_length does not have any effect?
            pass
        elif self.model_type == "vertex":
            model = GoogleGenAIEmbedding(
                model_name=self.model_name,
                vertexai_config={
                  "project": kwargs["project_id"],
                  "location": kwargs["region"],
                },
                embed_batch_size=100,
            )
        elif self.model_type == "vertex-endpoint":
            model = VertexEndpointEmbedding(
                endpoint_id=kwargs["embed_model_endpoint_id"],
                project_id=kwargs["project_id"],
                location=kwargs["region"],
                endpoint_kwargs={
                    "use_dedicated_endpoint": kwargs["embed_model_use_dedicated_endpoint"],
                },
            )  # max_length does not have any effect?
        elif self.model_type == "openai":
            # TODO - Add OpenAI embedding model
            # embed_model = OpenAIEmbedding()
            raise NotImplementedError

        return model


class RAGQueryEngine:
    """
    https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    TODO - Check other args for RetrieverQueryEngine
    """

    def __init__(self, retriever_type, vector_index):
        self.retriever_type = retriever_type
        self.index = vector_index
        self.retriever = None
        self.node_postprocessor = None
        self.response_synthesizer = None

    def create(self, similarity_top_k, response_mode, **kwargs):
        self.set_retriever(similarity_top_k, **kwargs)
        self.set_response_synthesizer(response_mode=response_mode)
        if kwargs["use_reranker"]:
            self.set_node_postprocessors(rerank_top_k=kwargs["rerank_top_k"])
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=self.node_postprocessor,
            response_synthesizer=self.response_synthesizer,
        )
        return query_engine

    def set_retriever(self, similarity_top_k, **kwargs):
        # Other retrievers can be used based on the type of index: List, Tree, Knowledge Graph, etc.
        # https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers.html
        # Find LlamaIndex equivalents for the following:
        # Check MultiQueryRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
        # Check Contextual compression from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
        # Check Ensemble Retriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
        # Check self-query from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
        # Check WebSearchRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
        if self.retriever_type == "vector_index":
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=kwargs["query_mode"],
                alpha=kwargs["hybrid_search_alpha"],
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                nodes=kwargs["nodes"],
                tokenizer=kwargs["tokenizer"],
                similarity_top_k=similarity_top_k,
            )
        else:
            raise NotImplementedError(
                f"Incorrect retriever type - {self.retriever_type}"
            )

    def set_node_postprocessors(self, rerank_top_k=2):
        # Node postprocessor: Porcessing nodes after retrieval before passing to the LLM for generation
        # Re-ranking step can be performed here!
        # Nodes can be re-ordered to include more relevant ones at the top: https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html

        self.node_postprocessor = [LLMRerank(top_n=rerank_top_k)]

    def set_response_synthesizer(self, response_mode):
        # Other response modes: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#configuring-the-response-mode
        qa_prompt_tmpl = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query while providing an explanation. "
            "If your answer is in favour of the query, end your response with 'yes' otherwise end your response with 'no'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)

        self.response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_prompt_tmpl,
            response_mode=response_mode,
        )
