from src.utils import get_db_profile, get_pixels_path, compute_embeddings
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pixelsparser
import chromadb
import logging
import base64
import os

load_dotenv()
MODEL = os.getenv("MODEL")
USE_OPENAI_MODEL = os.getenv("EMBEDDING_MODEL") == "text-embedding-3-small"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 50

os.makedirs("./database", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./database")
DB_PROFILE = get_db_profile(chroma_client)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # WARNINGS/INFO/DEBUG
logger.addHandler(logging.StreamHandler())




def create_embeddings():
    pixels_data = pixelsparser.load(get_pixels_path("./diary"))

    logger.info("Initialisation of ChromaDB collection...")

    items_to_embed = []
    collection = chroma_client.get_or_create_collection(name=f"pixels-rag-{DB_PROFILE}")
    for pixel in pixels_data:
        date = pixel.date.strftime("%Y-%m-%d")
        content = pixel.notes.strip()
        docId = base64.b64encode(date.encode()).decode()

        existing = collection.get(ids=[docId])
        if (content and not(existing["ids"])):
            items_to_embed.append((docId, date, content))

    if not(items_to_embed):
        logger.info("No new items to embed found.")
        return
    
    logger.info(f"{len(items_to_embed)} new items to embed found.")

    for i in tqdm(range(0, len(items_to_embed), BATCH_SIZE), desc="Computing embeddings"):
        batch = items_to_embed[i:i+BATCH_SIZE]
        doc_ids = [item[0] for item in batch]
        doc_data = [{"date": item[1], "content": item[2]} for item in batch]
        doc_contents = [item[2] for item in batch if item[2].strip()]

        try:
            embeddings = compute_embeddings(doc_contents)
            collection.add(ids=doc_ids, embeddings=embeddings, metadatas=doc_data)
        except Exception as e:
            logger.error(f"Error during embedding computation: {e}")
            continue




if __name__ == "__main__":
    create_embeddings()