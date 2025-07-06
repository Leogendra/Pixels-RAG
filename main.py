from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import chromadb
import base64

chroma_client = chromadb.PersistentClient(path="./")
# Load .env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def main():
    prompt = str(input("Prompt: "))

    logger.info("Converting prompt to embedding...")
    res = client.embeddings.create(input=prompt, model="text-embedding-3-small")
    embedding = res.data[0].embedding
    logger.info(f"Generated embedding for prompt: {embedding[:10]}...")

    logger.debug("Creating or getting the ChromaDB collection...")
    collection = chroma_client.get_or_create_collection(name="rag-experiment")
    logger.info("ChromaDB collection is ready.")

    logger.debug("Searching for similar documents in the collection...")
    res: chromadb.QueryResult = collection.query(
        query_embeddings=[embedding],
        n_results=5,
    )
    logger.info("Search results:")

    documentsToRetrieve = []
    for docIds in res["ids"][0]:
        logger.info(f"Document ID: {docIds}")
        decoded_ids = base64.b64decode(docIds).decode()
        documentsToRetrieve.append(decoded_ids)

    logger.info("Decoded Document paths")

    documentContent = []
    for doc in documentsToRetrieve:
        logger.info(f"Document Path: {doc}")
        with open(doc, "r") as file:
            content = file.read()
            logger.info("Document content retrieved")
            documentContent.append(content)
        


if __name__ == "__main__":
    main()
