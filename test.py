from dotenv import load_dotenv
import torch
import os


load_dotenv()
MODEL = os.getenv("MODEL")



if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")