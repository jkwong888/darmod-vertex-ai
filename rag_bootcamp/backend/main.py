
import os
import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager

# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

# from opentelemetry import trace
# from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID, SERVICE_NAME, Resource
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
# from opentelemetry.sdk import trace as trace_sdk
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine

from config import rag_cfg
from model import (
    LLMRequest,
    LLMResponse,
)

from rag_utils import (
    RAGLLM,
    RAGEmbedding,
    RAGQueryEngine,
    get_embed_model_dim,
    set_query_engine_args,
)
from storage_utils import RAGIndex




@asynccontextmanager
async def lifespan(app: FastAPI):
    # resource = Resource.create(attributes={
    #     # Use the PID as the service.instance.id to avoid duplicate timeseries
    #     # from different Gunicorn worker processes.
    #     SERVICE_INSTANCE_ID: f"worker-{os.getpid()}",
    # })
    # traceProvider = TracerProvider(resource=resource)
    # processor = BatchSpanProcessor(OTLPSpanExporter())
    # traceProvider.add_span_processor(processor)

    # LlamaIndexInstrumentor().instrument(tracer_provider=traceProvider)

    llm = RAGLLM(
        llm_type=rag_cfg['llm_type'],
        llm_name=rag_cfg['llm_name'],
        # api_base=GENERATOR_BASE_URL,
        # api_key=OPENAI_API_KEY,
    ).load_model(**rag_cfg)


    embed_model = RAGEmbedding(
        model_type=rag_cfg['embed_model_type'], 
        model_name=rag_cfg['embed_model_name']).load_model(**rag_cfg)

    Settings.llm = llm
    Settings.embed_model = embed_model
        
    index = await RAGIndex(
        db_type=rag_cfg['vector_db_type'],
        db_name=rag_cfg['vector_db_name'],
    ).load_index(**rag_cfg)


    query_engine_args = set_query_engine_args(rag_cfg)

    app.query_engine = RAGQueryEngine(
        retriever_type=rag_cfg['retriever_type'],
        vector_index=index,
    ).create(**query_engine_args)

    yield

    # TODO clean up weaviate connection


app = FastAPI(lifespan=lifespan)
app.query_engine = None



@app.post("/query")
async def query(llmReq: LLMRequest) -> LLMResponse:
    response = await app.query_engine.aquery(llmReq.query)

    return LLMResponse(
        response=response.response
    )

