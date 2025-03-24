# Source: https://learn.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/lesson/a0k5a/baseline-email-assistant
# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def save_png(png):
    # Write png to a file
    # Get path of current file
    current_file = os.path.abspath(__file__)
    # Get directory of current file
    current_dir = os.path.dirname(current_file)
    # Get path of current file
    file_path = os.path.join(current_dir, "email_agent.png")
    with open(file_path, "wb") as f:
        f.write(png)
