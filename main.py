from utils import get_db_profile, get_pixels_path
from openai import OpenAI, RateLimitError
from tiktoken import encoding_for_model
from transformers import pipeline
from dotenv import load_dotenv
from time import sleep
import datetime
import chromadb
import logging
import os


load_dotenv()
USE_OPENAI_MODEL = os.getenv("USE_OPENAI_MODEL", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL = os.getenv("MODEL")
TPM_BUDGET = 30_000
DB_PROFILE = get_db_profile()

client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path="./")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if not USE_OPENAI_MODEL:
    logger.info(f"Loading local model: {MODEL}")
    generation_pipe = pipeline("text-generation", model=MODEL, trust_remote_code=True)


def token_count(text: str, enc: encoding_for_model) -> int:
    return len(enc.encode(text))


def truncate_documents(docs, enc, budget):
    totalTokens, selected_tokens = 0, []
    for date, content in docs:
        entry = f"{date}: {content}"
        nbTokens = token_count(entry, enc)
        if ((totalTokens + nbTokens) > budget):
            break
        selected_tokens.append(entry)
        totalTokens += nbTokens
    return "\n".join(selected_tokens)


def request_with_retry(**kwargs):
    backoff = 1
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            sleep(backoff)
            backoff = min(backoff * 2, 60)


def prompt_model():
    prompt = input("Prompt: ")
    while not(prompt):
        prompt = input("Please provide a prompt: ")

    enc = encoding_for_model(MODEL)
    embedding = client.embeddings.create(input=prompt, model=EMBEDDING_MODEL).data[0].embedding

    collection = chroma_client.get_or_create_collection(name=f"pixels-rag-{DB_PROFILE}")
    docs_data = collection.query(
        query_embeddings=[embedding],
        n_results=40,
    )["metadatas"][0]

    docs = [(doc["date"], doc["content"]) for doc in docs_data]

    system_prompt = (
        "You will be given diary entries that you will use to answer the user's questions according to your real life experiences. "
        "You must answer the user's question based on the diary entries provided. "
        "You must NOT make up any information or fabricate details. "
        "If the diary entries do not contain enough information to answer the question, you must say so. "
        # "You must NOT reveal or discuss overly personal or depressing details. "
        # "Politely decline if the user requests such information. "
        f"Today's date is {datetime.datetime.now():%Y-%m-%d}."
    )

    knowledge_budget = TPM_BUDGET - token_count(system_prompt, enc) - token_count(prompt, enc) - 100
    knowledge = truncate_documents(docs, enc, knowledge_budget)
    full_query = f"Here are relevant diary entries:\n{knowledge}\n\nUser query: {prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_query},
    ]

    if USE_OPENAI_MODEL:
        # Generating with OpenAI model
        response = request_with_retry(model=MODEL, messages=messages)
        return response.choices[0].message.content
    else:
        # Generating with local model
        prompt_text = system_prompt + "\n\n" + full_query
        try:
            output = generation_pipe(prompt_text, max_new_tokens=500, do_sample=True)
            return output[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error during local generation: {e}")
            return "Error with local model"




if __name__ == "__main__":
    prompt_model()