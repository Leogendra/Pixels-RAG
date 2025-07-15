from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI
import pixelsparser
import numpy as np
import chromadb
import logging
import base64
import os


load_dotenv()
USE_OPENAI_MODEL = os.getenv("USE_OPENAI_MODEL", "").lower() == "true"
MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 50

chroma_client = chromadb.PersistentClient(path="./")
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # DEBUG to see more logs
logger.addHandler(logging.StreamHandler())
logging.getLogger("chromadb").setLevel(logging.WARNING)




def compute_embeddings(texts, embed_pipeline=None):
    if USE_OPENAI_MODEL:
        client = OpenAI(api_key=OPENAI_API_KEY)
        res = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [r.embedding for r in res.data]
    else:
        results = embed_pipeline(texts, truncation=True, padding=True)
        return [np.mean(r, axis=0).tolist() for r in results]


def load_local_embed_pipeline():
    logger.info(f"Downloading embedding model {EMBEDDING_MODEL} if not already present...")
    return pipeline("feature-extraction", model=EMBEDDING_MODEL)


def main():
    logger.info("Initialisation of ChromaDB collection...")
    collection = chroma_client.get_or_create_collection(name="pixels-rag")

    embed_pipeline = None
    if not(USE_OPENAI_MODEL):
        embed_pipeline = load_local_embed_pipeline()

    diary_data = pixelsparser.load("./diary/pixels.json")
    items_to_embed = []
    for pixel in diary_data:
        date = pixel.date.strftime("%Y-%m-%d")
        content = pixel.notes
        doc_id = base64.b64encode(date.encode()).decode()

        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        items_to_embed.append((doc_id, date, content))

    logger.info(f"{len(items_to_embed)} new items to embed found.")

    for i in range(0, len(items_to_embed), BATCH_SIZE):
        batch = items_to_embed[i:i + BATCH_SIZE]
        ids = [item[0] for item in batch]
        metadatas = [{"date": item[1], "content": item[2]} for item in batch]
        contents = [item[2] for item in batch]

        logger.info(f"-> Batch {i//BATCH_SIZE}/{len(items_to_embed)//BATCH_SIZE}...")

        try:
            embeddings = compute_embeddings(contents, embed_pipeline=embed_pipeline)
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logger.info("   Insertion successful.")
        except Exception as e:
            logger.error(f"Error during embedding computation: {e}")
            continue




if __name__ == "__main__":
    main()
