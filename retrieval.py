from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
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
    logger.info("Starting the retrieval process...")

    logger.debug("Creating or getting the ChromaDB collection...")
    collection = chroma_client.get_or_create_collection(
        name="rag-experiment",
    )
    logger.info("ChromaDB collection is ready.")

    logger.debug("Recursively reading files from the content directory...")
    # Recursively read all files in the content directory
    content_directory = ".\\content"
    for root, dirs, files in os.walk(content_directory):
        for file in files:
            file_path = os.path.join(root, file)
            logger.debug(f"Processing file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.debug(f"Read content from {file_path}")
                    res = client.embeddings.create(
                        input=content, model="text-embedding-3-small"
                    )

                    embedding = res.data[0].embedding
                    logger.info(
                        f"Generated embedding for {file_path}: {embedding[:10]}..."
                    )
                    logger.debug("Inserting embedding into the ChromaDB collection...")
                    collectionId = base64.b64encode(file_path.encode()).decode()
                    collection.add(
                        ids=[collectionId],
                        embeddings=[embedding],
                    )

            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue


if __name__ == "__main__":
    main()
