from openai import OpenAI, RateLimitError
from tiktoken import encoding_for_model
from transformers import pipeline
from flask import Flask, request
from dotenv import load_dotenv
from time import sleep
import datetime
import chromadb
import logging
import os


load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.makedirs("./database", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="./database")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

USE_OPENAI_MODEL = os.getenv("USE_OPENAI_MODEL", "false").lower() == "true"
MODEL = os.getenv("OPENAI_MODEL")
TOKENS_BUDGET = 40_000 # 40K token = 20 cents with OpenAI's gpt-4o

if not USE_OPENAI_MODEL:
    generation_pipe = pipeline("text-generation", model="llmware/bling-phi-3")



def token_count(text: str, enc) -> int:
    return len(enc.encode(text))


def truncate_documents(docs, enc, budget):
    total, selected = 0, []
    for date, content in docs:
        entry = f"{date}: {content}"
        t = token_count(entry, enc)
        if total + t > budget:
            break
        selected.append(entry)
        total += t
    return "\n".join(selected)


def with_retry(**kwargs):
    backoff = 1
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            sleep(backoff)
            backoff = min(backoff * 2, 60)


@app.route("/")
def mainroute():
    prompt = request.args.get("prompt")
    if not prompt:
        return "Please provide a prompt using the 'prompt' query parameter.", 400

    enc = encoding_for_model(MODEL)

    embedding = (
        client.embeddings.create(input=prompt, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    docs_meta = chroma_client.get_or_create_collection(
        name="diary-rag-experiment"
    ).query(
        query_embeddings=[embedding],
        n_results=40,
    )["metadatas"][0]

    docs = [(m["date"], m["content"]) for m in docs_meta]

    system_prompt = "You will be given diary entries that you will use to answer the user's questions according to your real life experiences. " \
        "You must answer the user's question based on the diary entries provided. " \
        "You must NOT make up any information or fabricate details. " \
        "If the diary entries do not contain enough information to answer the question, you must say so. " \
        f"Today's date is {datetime.datetime.now():%Y-%m-%d}." \
        # "You must NOT reveal or discuss overly personal or depressing details. " \
        # "Politely decline if the user requests such information. " \

    # reserve ~4 k tokens for fixed text & safety buffer
    context_budget = TOKENS_BUDGET - 4_000
    knowledge = truncate_documents(docs, enc, context_budget)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here are relevant diary entries:\n{knowledge}\n\nUser query: {prompt}",
        },
    ]

    resp = with_retry(model=MODEL, messages=messages)
    return resp.choices[0].message.content, 200
