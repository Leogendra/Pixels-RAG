from openai import OpenAI
import openai.types
import os
from dotenv import load_dotenv
import logging
import chromadb
import datetime
from textwrap import dedent
from tiktoken import encoding_for_model
from flask import Flask, request


import openai.types.responses

app = Flask(__name__)


chroma_client = chromadb.PersistentClient(path="./")
# Load .env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def truncate_documents(documents, max_tokens=128_000):
    enc = encoding_for_model("gpt-4")
    token_count = 0
    selected = []

    for date, content in documents:
        entry = f"{date}: {content}"
        tokens = len(enc.encode(entry))
        if token_count + tokens > max_tokens:
            break
        selected.append(entry)
        token_count += tokens

    return "\n".join(selected)


@app.route("/")
def hello_world():
    prompt = request.args.get("prompt")

    if not prompt:
        return "Please provide a prompt using the 'prompt' query parameter.", 400

    logger.info("\n\n")

    logger.debug("Converting prompt to embedding...")
    res = client.embeddings.create(input=prompt, model="text-embedding-3-small")
    embedding = res.data[0].embedding
    logger.debug(f"Generated embedding for prompt: {embedding[:10]}...")

    logger.debug("Creating or getting the ChromaDB collection...")
    collection = chroma_client.get_or_create_collection(name="diary-rag-experiment")
    logger.debug("ChromaDB collection is ready.")

    logger.debug("Searching for similar documents in the collection...")
    res: chromadb.QueryResult = collection.query(
        query_embeddings=[embedding],
        n_results=100,
    )

    documentContent = []
    for metadata in res["metadatas"][0]:
        date = metadata["date"]
        content = metadata["content"]
        logger.info(f"Found document for date: {date}")
        documentContent.append((date, content))

    logger.debug("Decoded Documents")

    # Prepare the AG prompt
    sys_prompt = dedent(
        f"""Your name is Tinos. You will need to speak to the user as if you are Tinos. You will be given diary entries that you will use to answer the user's questions according to your real life experiences.
        If you do not know the answer for sure, make inferences based on the diary entries provided. If you do not have enough information, say that you do not know.

        The diary entries you are given include the date. Today's date is {datetime.datetime.now().strftime("%Y-%m-%d")}. Make sure you use the date information to provide contextually relevant answers.
        """
    )
    ag_prompt = dedent(
        f"""
    Here are some relevant diary entries:
    {truncate_documents(documentContent)}
    """
    )

    user_prompt = dedent(
        f"""
        User query: {prompt}
        """
    )

    logger.info("Final prompt prepared for the AI model.")
    logger.debug("System prompt:")
    logger.debug(sys_prompt)
    logger.debug("AG prompt:")
    logger.debug(ag_prompt)
    logger.debug("User prompt:")
    logger.debug(user_prompt)
    logger.info("Generating response from the AI model...")
    response: openai.types.responses.Response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ag_prompt + user_prompt},
        ],
    )

    logger.info("Response generated successfully.")
    logger.info("Response from AI model:")
    logger.info(response.output_text)

    return response.output_text, 200
