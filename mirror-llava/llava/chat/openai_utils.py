import os
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

class OpenaiGpt:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def request(self, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,  
            messages=kwargs['messages'],
            temperature=kwargs['temperature'],
            max_tokens=kwargs['max_tokens'],
            top_p=kwargs['top_p'],
            frequency_penalty=kwargs['frequency_penalty'],
            presence_penalty=kwargs['presence_penalty'],
            stop=kwargs['stop']
        )
        return {
            "content": response['choices'][0]['message']['content'],
            "usage": response['usage'],
            "model": response['model']
        }
