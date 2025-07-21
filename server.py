from src.utils import get_db_profile, compute_embeddings
from transformers import pipeline, AutoTokenizer
from openai import OpenAI, RateLimitError
from tiktoken import encoding_for_model
from flask import Flask, request
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
TOKENS_BUDGET = 40_000 if USE_OPENAI_MODEL else 4_000 # 40K token = 20 cents with OpenAI's gpt-4o
NUMBER_OF_ENTRIES = 40

app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs("./database", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./database")
DB_PROFILE = get_db_profile(chroma_client)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())




def truncate_entries(entries: list, enc: encoding_for_model, budget: int) -> str:
    totalTokens, selected_tokens = 0, []
    entries.sort(key=lambda x: x[0], reverse=True)  # Sort by date ascending to make sure recent entries are prioritized
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
        n_results=NUMBER_OF_ENTRIES,
    )["metadatas"][0]

    entries = [(entry["date"], entry["content"]) for entry in entries_data]

    system_prompt = "You will be given diary entries that you will use to answer the user's questions. " \
        "You must base your answers on the diary entries provided, and you are allowed to infer reasonable conclusions from them, " \
        "even if the answer is not explicitly written. " \
        "Do not invent facts unrelated to the entries, but feel free to interpret, connect, and deduce information from them. " \
        "If truly no reasonable inference can be made, you can say so. " \
        "You must answer in the same language as the user's question. " \
        "You must not mention the diary entries in your answer, just use them to inform your response. " \
        f"Today's date is {datetime.datetime.now():%Y-%m-%d}." \

    knowledge_budget = TOKENS_BUDGET - count_token(system_prompt, encoder) - count_token(prompt, encoder) - 20 # Text below
    knowledge = truncate_entries(entries, encoder, knowledge_budget)
    full_query = f"Here are relevant diary entries:\n{knowledge}\n\nUser query: {prompt}"

    if USE_OPENAI_MODEL:
        response = request_with_retry(model=MODEL, messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_query},
        ])
        return response.choices[0].message.content if response.choices else "No response from OpenAI"
    else:
        prompt_text = f"{system_prompt}\n\n{full_query}"
        try:
            generation_model = pipeline("text-generation", model=MODEL, device=DEVICE)
            output = generation_model(
                prompt_text,
                max_new_tokens=500,
                do_sample=True,
                top_p=0.9,
            )
            return output[0]["generated_text"][len(prompt_text):].strip()
        except Exception as e:
            logger.error(f"Error during local generation: {e}")
            return "Error with local model"


@app.route("/")
def prompt_model():
        prompt = request.args.get("prompt")

        if not(prompt.strip()):
            return "Please provide a prompt using the 'prompt' query parameter.", 400

        return infer_with_model(prompt).replace("\n", " "), 200
