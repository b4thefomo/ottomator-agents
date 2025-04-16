import os
import asyncio
from typing import Any, Optional
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import httpx

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "neo4j+s://aa17251b.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "8e09oKdx2zq2e5WLUR5xPmB88GFk7CW36yhowVod8Uk"
os.environ["NEO4J_DATABASE"] = "neo4j" # Explicitly use the default database

# Load environment variables from .env file
dotenv.load_dotenv()

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# URL of the Pydantic AI documentation
PYDANTIC_DOCS_URL = "https://ai.pydantic.dev/llms.txt"

def fetch_pydantic_docs() -> str:
    """Fetch the Pydantic AI documentation from the URL.
    
    Returns:
        The content of the documentation
    """
    try:
        response = httpx.get(PYDANTIC_DOCS_URL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching Pydantic AI documentation: {e}")


async def async_main():
    # Initialize RAG instance and insert Pydantic documentation
    print("Initializing RAG system...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
    )

    print("Initializing storage...")
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    print("Fetching documentation...")
    docs = fetch_pydantic_docs()
    
    print("Inserting documentation into RAG system...")
    await rag.ainsert(docs)  # Use ainsert instead of insert
    
    print("Successfully inserted Pydantic documentation into the RAG system.")
    return rag


def main():
    print("Starting Pydantic documentation ingestion process...")
    asyncio.run(async_main())
    print("Done! The RAG system is ready to use.")


if __name__ == "__main__":
    main()