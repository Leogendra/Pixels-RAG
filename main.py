from src.utils import get_db_profile, compute_embeddings
from transformers import pipeline, AutoTokenizer
from openai import OpenAI, RateLimitError
from tiktoken import encoding_for_model
from dotenv import load_dotenv
from time import sleep
import torch
import datetime
import chromadb
import logging
import os


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
USE_OPENAI_MODEL = EMBEDDING_MODEL == "text-embedding-3-small"
MODEL = os.getenv("MODEL")
DEVICE = 0 if torch.cuda.is_available() else -1
TOKENS_BUDGET = 40_000 # 40K token = 20 cents with OpenAI's gpt-4o

client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs("./database", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./database")
DB_PROFILE = get_db_profile(chroma_client)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())




def truncate_entries(entries: list, enc: encoding_for_model, budget: int) -> str:
    totalTokens, selected_tokens = 0, []
    for date, content in entries:
        entry = f"{date}: {content}"
        nbTokens = count_token(entry, enc)
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
            backoff = min(backoff + 5, 60)
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            return None


def get_encoder():
    if USE_OPENAI_MODEL:
        return encoding_for_model(MODEL)
    else:
        return AutoTokenizer.from_pretrained(MODEL)


def count_token(text: str, encoder) -> int:
    return len(encoder.encode(text))


def infer_with_model(prompt: str) -> str:
    encoder = get_encoder()
    embedding = compute_embeddings([prompt])[0]

    collection = chroma_client.get_or_create_collection(name=f"pixels-rag-{DB_PROFILE}")
    entries_data = collection.query(
        query_embeddings=[embedding],
        n_results=40,
    )["metadatas"][0]

    entries = [(entry["date"], entry["content"]) for entry in entries_data]

    system_prompt = "You will be given diary entries that you will use to answer the user's questions according to your real life experiences. " \
        "You must answer the user's question based on the diary entries provided. " \
        "You must NOT make up any information or fabricate details. " \
        "If the diary entries do not contain enough information to answer the question, you must say so. " \
        f"Today's date is {datetime.datetime.now():%Y-%m-%d}."

    knowledge_budget = TOKENS_BUDGET - count_token(system_prompt, encoder) - count_token(prompt, encoder) - 100
    knowledge = truncate_entries(entries, encoder, knowledge_budget)
    full_query = f"Here are relevant diary entries:\n{knowledge}\n\nUser query: {prompt}"


    if USE_OPENAI_MODEL:
        response = request_with_retry(model=MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_query},
        ])
        return response.choices[0].message.content
    else:
        prompt_text = f"{system_prompt}\n\n{full_query}"
        try:
            generation_model = pipeline("text-generation", model=MODEL, device=device)
            output = generation_model(
                prompt_text,
                max_new_tokens=500,
                do_sample=True,
                top_p=0.9,
            )
            return output[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error during local generation: {e}")
            return "Error with local model"





if __name__ == "__main__":
    prompt = input("Prompt: ")
    while not(prompt):
        prompt = input("Please provide a prompt: ")

    response = infer_with_model(prompt)

    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)

    print(f"Response saved to ./response.txt")