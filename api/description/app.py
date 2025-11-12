import os
from flask import Flask, request, jsonify

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts import build_message_for_llama

app = Flask(__name__)
model_name_or_path = "/home/model/Meta-Llama-3-8B-Instruct"

print(f"Loading Model...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
llm = LLM(
    model=model_name_or_path, 
    tensor_parallel_size=1,
    device="cuda"
)
sampling_params = SamplingParams(
    temperature=0.7,     
    top_p=0.9,           
    max_tokens=128       
)

@app.route('/describe', methods=['POST'])
def verify_image():
    if 'history' not in request.form or 'client_utt' not in request.form:
        return jsonify({"error": "Missing required fields"}), 400

    history = request.form['history']
    client_utt = request.form['client_utt']
    messages = build_message_for_llama(dialogue=history, utter=client_utt)
    prompts_chat_applied = [tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )]
    
    outputs = llm.generate(prompts_chat_applied, sampling_params)
    generated_texts = [o.outputs[0].text for o in outputs]
    try:
        history = request.form['history']
        client_utt = request.form['client_utt']
        messages = build_message_for_llama(dialogue=history, utter=client_utt)
        prompts_chat_applied = [tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )]
        
        outputs = llm.generate(prompts_chat_applied, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]
        return jsonify({"response": generated_texts[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Image Verifier API is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
