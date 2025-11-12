import openai
API_KEY= ""
openai.api_key = API_KEY

def chatgpt_request(messages, model):
    response = openai.ChatCompletion.create(
        model=model,  
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.2,
        stop=None
    )
    return {
        "content": response['choices'][0]['message']['content'],
        "usage": response['usage'],
        "model": response['model']
    }