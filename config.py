import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")


def get_llm():
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
