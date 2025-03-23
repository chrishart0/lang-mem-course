# Source: https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/lesson/a0k5a/baseline-email-assistant

from helper import get_openai_api_key

openai_api_key = get_openai_api_key()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Basic chat test
print(llm.invoke("What is the capital of France?"))



