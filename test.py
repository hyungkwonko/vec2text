import os
from dotenv import load_dotenv
from openai import OpenAI

import torch
import vec2text

load_dotenv(".env")
MODEL = "text-embedding-ada-002"


def get_embeddings_openai(model=MODEL):
    outputs = []
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input="Jack Morris is a PhD student at Cornell Tech in New York City",
        model=model,
        encoding_format="float",
    )
    outputs.extend([response.data[0].embedding])
    return torch.tensor(outputs)


embeddings = get_embeddings_openai()
corrector = vec2text.load_corrector(MODEL)
text = vec2text.invert_embeddings(embeddings=embeddings.to("mps"), corrector=corrector)
print(text[0])
