
import weaviate

from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from config import get_weaviate_api_key


class RAGIndex:
    """
    Use storage context to set custom vector store
    Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    """

    def __init__(self, db_type, db_name):
        self.db_type = db_type
        self.db_name = db_name
        self._persist_dir = f"./.{db_type}_index_store/"

    async def load_index(self, save=True, **kwargs):
        # Only supports Weaviate as of now
        if self.db_type == "weaviate":
            # with open(Path.home() / ".weaviate.key", "r") as f:
            #     weaviate_api_key = f.read().rstrip("\n")
            weaviate_client = weaviate.use_async_with_weaviate_cloud(
                cluster_url=kwargs["weaviate_url"],
                auth_credentials=weaviate.auth.AuthApiKey(get_weaviate_api_key()),
            )

            vector_store = WeaviateVectorStore(
                weaviate_client=weaviate_client,
                index_name=self.db_name,
            )

            print(f"Connecting to weaviate {kwargs['weaviate_url']}")
            await weaviate_client.connect()

        # jkwng: added Vertex AI Vector Search support here
        elif self.db_type == "vertex":
            pass

        #   # setup storage
        #   vector_store = VertexAIVectorStore(
        #       project_id=PROJECT_ID,
        #       region=REGION,
        #       index_id=vs_index.resource_name,
        #       endpoint_id=vs_endpoint.resource_name,
        #       gcs_bucket_name=dst_bucket,
        #   )

        else:
            raise NotImplementedError(f"Incorrect vector db type - {self.db_type}")

        # Re-index
        print("Loading index ...")
        index = VectorStoreIndex.from_vector_store(vector_store)

        return index
