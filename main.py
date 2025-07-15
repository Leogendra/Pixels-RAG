from transformers import pipeline
from dotenv import load_dotenv
from textwrap import dedent
import numpy as np
import chromadb
import logging
import base64
import openai
import os


load_dotenv()
USE_OPENAI_MODEL = os.getenv("USE_OPENAI_MODEL", "").lower() == "true"
MODEL = os.getenv("MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

chroma_client = chromadb.PersistentClient(path="./")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())




def get_embedding(text):
    if USE_OPENAI_MODEL:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        res = openai.Embedding.create(input=text, model=MODEL)
        return res.data[0].embedding
    else:
        embed_pipeline = pipeline("feature-extraction", model=EMBEDDING_MODEL)
        vectors = embed_pipeline(text, truncation=True, padding=True)[0]
        return np.mean(vectors, axis=0).tolist()


def generate_response(system_prompt, user_prompt):
    if USE_OPENAI_MODEL:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message["content"]
    else:
        hf_model = pipeline("text-generation", model=MODEL)
        prompt = system_prompt + "\n" + user_prompt
        return hf_model(prompt, max_new_tokens=250)[0]["generated_text"]


def main():
    prompt = str(input("Prompt: "))
    logger.info("Generating embedding...")
    embedding = get_embedding(prompt)
    logger.info(f"Embedding: {embedding[:10]}...")

    collection = chroma_client.get_or_create_collection(name="pixels-rag")
    res = collection.query(query_embeddings=[embedding], n_results=5)

    documentsToRetrieve = [base64.b64decode(docId).decode() for docId in res["ids"][0]]
    documentContent = []
    for doc in documentsToRetrieve:
        with open(doc, "r", encoding="utf-8") as f:
            documentContent.append(f.read())

    sys_prompt = dedent(
        """You are an AI assistant for the React documentation. Keep answers concise and precise."""
    )
    ag_prompt = dedent(
        f"""Here are relevant documents:\n{"\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documentContent)])}"""
    )
    user_prompt = dedent(f"""User query: {prompt}""")

    logger.info("Generating final response...")
    final_response = generate_response(sys_prompt, ag_prompt + "\n" + user_prompt)
    logger.info("AI Response:")
    logger.info(final_response)




if __name__ == "__main__":
    main()
