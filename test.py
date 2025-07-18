from transformers import AutoTokenizer
from dotenv import load_dotenv
import os


load_dotenv()
MODEL = os.getenv("MODEL")



if __name__ == "__main__":
    text = "jaaaj"
    encoder = AutoTokenizer.from_pretrained(MODEL)
    res1 = encoder.encode(text)
    print(res1)
    print(len(res1))
