from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import torch
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
USE_OPENAI_MODEL = EMBEDDING_MODEL == "text-embedding-3-small"
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


def get_db_profile(chroma_client: chromadb.PersistentClient) -> str:
    existing_collections = chroma_client.list_collections()
    db_profiles = [collection.name.replace("pixels-rag-", "") for collection in existing_collections if collection.name.startswith("pixels-rag-")]
    print("")

    if not(db_profiles):
        dbProfile = input("Enter a profile name for the database (e.g. Bob): ").strip()
    else:
        print("Available profiles:")
        for i, profile in enumerate(db_profiles):
            print(f"{i + 1}. {profile.strip()}")

        choice = input("Select a profile by number or enter a new one: ")
        if ("del" in choice.lower()):
            nbProfile = input("Select the profile number to delete (0 to exit): ").strip()
            if (nbProfile.isdigit() and (1 <= int(nbProfile) <= len(db_profiles))):
                profile_to_delete = db_profiles[int(nbProfile) - 1]
                chroma_client.delete_collection(name=f"pixels-rag-{profile_to_delete}")
                print(f"Profile '{profile_to_delete}' deleted.")
                return get_db_profile(chroma_client)
            else:
                print("No profile deleted.")
                return get_db_profile(chroma_client)
        elif (choice.isdigit() and (1 <= int(choice) <= len(db_profiles))):
            return db_profiles[int(choice) - 1].strip()
        else:
            dbProfile = choice.strip()

    return f"{dbProfile}-{EMBEDDING_MODEL.replace('/', '-')}"


def get_pixels_path(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    
    if len(files) == 1:
        return os.path.join(folder, files[0])
    
    print("Available JSON files:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    choice = ""
    while not(choice.isdigit() and (1 <= int(choice) <= len(files))):
        choice = input("Select a file by number: ")

    return os.path.join(folder, files[int(choice) - 1])