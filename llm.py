import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen/qwen3.7-flash"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_retries=3,
        extra_body={"reasoning": {"enabled": False}},  # qwen3.7-flash: 12.7s -> 1.1s per call, ~50x fewer output tokens
    )
