from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
import chromadb
import base64

import pixelsparser

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
    collection: chromadb.Collection = chroma_client.get_or_create_collection(
        name="diary-rag-experiment",
    )
    logger.info("ChromaDB collection is ready.")

    logger.debug("Recursively reading files from the content directory...")

    diary_data = pixelsparser.load(".\\diary\\pixels.json")
    for pixel in diary_data:
        date = pixel.date.strftime("%Y-%m-%d")
        content = pixel.notes
        logger.debug(f"Processing diary entry for date: {date}")
        try:
            logger.debug(f"Converting content to embedding for date: {date}")
            res = client.embeddings.create(
                input=content, model="text-embedding-3-small"
            )

            embedding = res.data[0].embedding
            logger.info(f"Generated embedding for {date}: {embedding[:10]}...")
            logger.debug("Inserting embedding into the ChromaDB collection...")
            collectionId = base64.b64encode(date.encode()).decode()
            collection.add(
                ids=[collectionId],
                embeddings=[embedding],
                metadatas=[{"date": date, "content": content}],
            )

        except Exception as e:
            logger.error(f"Error processing diary entry for {date}: {e}")
            continue


if __name__ == "__main__":
    main()
