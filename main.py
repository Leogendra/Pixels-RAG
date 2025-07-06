from openai import OpenAI
import openai.types
import os
from dotenv import load_dotenv
import logging
import chromadb
import base64
from textwrap import dedent

import openai.types.responses

chroma_client = chromadb.PersistentClient(path="./")
# Load .env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def main():
    prompt = str(input("Prompt: "))

    logger.info("Converting prompt to embedding...")
    res = client.embeddings.create(input=prompt, model="text-embedding-3-small")
    embedding = res.data[0].embedding
    logger.info(f"Generated embedding for prompt: {embedding[:10]}...")

    logger.debug("Creating or getting the ChromaDB collection...")
    collection = chroma_client.get_or_create_collection(name="rag-experiment")
    logger.info("ChromaDB collection is ready.")

    logger.debug("Searching for similar documents in the collection...")
    res: chromadb.QueryResult = collection.query(
        query_embeddings=[embedding],
        n_results=5,
    )
    logger.info("Search results:")

    documentsToRetrieve = []
    for docIds in res["ids"][0]:
        logger.info(f"Document ID: {docIds}")
        decoded_ids = base64.b64decode(docIds).decode()
        documentsToRetrieve.append(decoded_ids)

    logger.info("Decoded Document paths")

    documentContent = []
    for doc in documentsToRetrieve:
        logger.info(f"Document Path: {doc}")
        with open(doc, "r", encoding="utf-8") as file:
            content = file.read()
            logger.info("Document content retrieved")
            documentContent.append(content)

    # Prepare the AG prompt
    sys_prompt = dedent(
        """You are an AI assistant for the React documentation. You must help users with their queries. Keep information very short and straight to the point. No waffle."""
    )
    ag_prompt = dedent(
        f"""
        Here are some relevant documents from the React documentation:
        {"\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(documentContent)])}
        """
    )

    user_prompt = dedent(
        f"""
        User query: {prompt}
        """
    )

    logger.info("Final prompt prepared for the AI model.")
    logger.info("Generating response from the AI model...")
    response: openai.types.responses.Response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": ag_prompt + user_prompt},
        ],
    )

    logger.info("Response generated successfully.")
    logger.info("Response from AI model:")
    logger.info(response.output_text)


if __name__ == "__main__":
    main()
