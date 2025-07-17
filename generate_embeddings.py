from utils import get_db_profile, get_pixels_path
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pixelsparser
import chromadb
import logging
import base64
import torch
import os

load_dotenv()
USE_OPENAI_MODEL = os.getenv("USE_OPENAI_MODEL", "").lower() == "true"
MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 50

chroma_client = chromadb.PersistentClient(path="./")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # WARNINGS/INFO/DEBUG
logger.addHandler(logging.StreamHandler())
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)




def compute_embeddings(texts_list):
    if USE_OPENAI_MODEL:
        client = OpenAI(api_key=OPENAI_API_KEY)
        result = client.embeddings.create(input=texts_list, model=EMBEDDING_MODEL)
        return [res.embedding for res in result.data]
    else:
        tokens = tokenizer(
            texts_list,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.tolist()


def create_embeddings():
    pixels_data = pixelsparser.load(get_pixels_path("./diary"))

    logger.info("Initialisation of ChromaDB collection...")
    profileName = get_db_profile()
    
    items_to_embed = []
    collection = chroma_client.get_or_create_collection(name=f"pixels-rag-{profileName}")
    for pixel in pixels_data:
        date = pixel.date.strftime("%Y-%m-%d")
        content = pixel.notes
        docId = base64.b64encode(date.encode()).decode()

        existing = collection.get(ids=[docId])
        if not(existing["ids"]):
            items_to_embed.append((docId, date, content))

    if not(items_to_embed):
        logger.info("No new items to embed found.")
        return
    
    logger.info(f"{len(items_to_embed)} new items to embed found.")

    for i in tqdm(range(0, len(items_to_embed), BATCH_SIZE), desc="Computing embeddings"):
        batch = items_to_embed[i:i+BATCH_SIZE]
        doc_ids = [item[0] for item in batch]
        doc_metadatas = [{"date": item[1], "content": item[2]} for item in batch]
        doc_contents = [item[2] for item in batch]

        try:
            embeddings = compute_embeddings(doc_contents)
            collection.add(ids=doc_ids, embeddings=embeddings, metadatas=doc_metadatas)
        except Exception as e:
            logger.error(f"Error during embedding computation: {e}")
            continue




if __name__ == "__main__":
    create_embeddings()
