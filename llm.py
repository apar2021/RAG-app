from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
import requests
from llama_index.core import Settings
from typing import Optional, List, Mapping, Any
import subprocess
import shlex

def call_llm(model_path, **additional_kwargs):
    


class Model(CustomLLM):
    api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    def __init__(self.api_url):
        super().init()
        self.api_url = api_url
