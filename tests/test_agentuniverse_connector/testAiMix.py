import os

import requests
import time
import json
import time

from langchain.llms import OpenAI

API_SECRET_KEY = "sk-pOkyqQLHsRTIdabl6e556aB2A4B44bFeBb110aAcB9Ea2052"
BASE_URL = "https://aihubmix.com/V1"  # aihubmix的base-url

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL


def text():
    llm = OpenAI(temperature=0.1)
    text = "你是谁？"
    print(llm(text))


if __name__ == '__main__':
    text()
