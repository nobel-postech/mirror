import os
import json
import openai
from openai import OpenAI

openai.api_key = os.environ['OPENAI_API_KEY']
class OpenaiGpt:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )
        
    def request(self, **kwargs):
        chat_completion = self.client.chat.completions.create(
            messages=kwargs['messages'],
            model=self.model_name,
            temperature=kwargs['temperature'],
            max_tokens=kwargs['max_tokens'],
            top_p=kwargs['top_p'],
            frequency_penalty=kwargs['frequency_penalty'],
            presence_penalty=kwargs['presence_penalty'],
            stop=kwargs['stop']
        )
        response = json.loads(chat_completion.json())
        return {
            "content": response['choices'][0]['message']['content'],
            "usage": response['usage'],
            "model": response['model']
        }