from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_API_TOKEN")
)

image = client.text_to_image(
    "A cyberpunk cat wearing sunglasses",
    model="black-forest-labs/FLUX.1-schnell"
)

image.save("test.png")
print("Success")