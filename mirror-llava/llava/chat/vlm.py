import os
import re
import torch
import pandas as pd

from typing import List
from PIL import Image
from pathlib import Path
from os.path import join as pjoin

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates

from llava import conversation as conversation_lib
from llava.chat.data_utils import *

current_dir  = Path(__file__).parent

IMAGE_DESC = "<image>\nThe image above shows the client."
IMG_EMOTION_RECOG_DESC = "Look at the provided image and assess the client’s emotional state. Clearly describe their emotions in simple, phase-based steps for easy understanding."
CBT_DESC = "Based on their body language and facial expression, respond as a psychotherapist conducting a CBT (Cognitive Behavioral Therapy) session."

PLANNING_DESC = """Choose an appropriate CBT technique and create a counseling plan based on that technique. 

Respond in the following format:

CBT technique:
{{selected_cbt}}

Counseling planning:
{{generated_cbt_plan}}"""

class Llava:
    def __init__(self, args) -> None:
        self.model_path = args.model_path
        self.model_base = args.model_base
        self.conv_mode = args.conv_mode
        
        self._load_model()
        
        conv = conversation_lib.default_conversation.copy()
        assert conversation_lib.default_conversation.version.startswith("v1"), conversation_lib.default_conversation.version

        self.roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    def _load_model(self):
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.vlm, self.image_processor, self.context_len = load_pretrained_model(
            model_path, self.model_base, model_name)
        
    def generate(self, prompt, image_tensor, image, **kwargs):
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        default_params = {
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
            "stop": None,
            "num_beams": 1,
        }
        params = {**default_params, **kwargs}
        with torch.inference_mode():
            output_ids = self.vlm.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if params['temperature'] > 0 else False,
                        temperature=params['temperature'],
                        top_p=params['top_p'],
                        num_beams=1,
                        max_new_tokens=1024,
                        use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
class Counselor(Llava):
    def __init__(
        self, 
        args):
        super().__init__(args=args)
        self.ctx_len = args.ctx_len

    def _add_to_history(self, role, message):
        self.history.append({"role": role, "content": message})

    def set_client(self, client_information, reason_counseling):
        self.client_information = client_information
        self.reason_counseling = reason_counseling
        self.history = []
        self.cbt_plan = None
    
    def _get_history_text(self, history: List[str], ctx_len: int=None):
        if ctx_len:
            history = history[-ctx_len:]
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in history
            ]
        )
        return history_text
    
    def build_prompt(self, prompt, image_file):
        conv = conv_templates[self.conv_mode].copy()
        role = self.roles["human"]
        
        if "<image>" in prompt:
            if self.vlm.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt.replace("<image>", "")
        
        conv.append_message(role, prompt)
        conv.append_message(conv.roles[1], None)  
        prompt = conv.get_prompt()
                
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.vlm.config)[0]
        return prompt, image_tensor, image
    
    def post_process_response(self, response):
        if "'message':" in response:
            response = self.clean_message(response)
        response = self.extract_counselor_message(response)
        return response.strip()

    def extract_after_title(self, text):
        result = re.split(r"(Counselor|Psychotherapist|Therapist)\s*:", text)
        return result[-1] if result else text

    def extract_counselor_message(self, response):
        response = self.extract_after_title(response)
        for delimiter in ["Client :", "Client:"]:
            if delimiter in response:
                response = response.split(delimiter)[0]

        response = response.replace("\n", "")
        response = response.replace("\\", "")
        response = response.replace('"', "")
        return response


class LlavaMirrorCounselor(Counselor):
    def __init__(
        self, 
        args, 
    ):
        super().__init__(args=args)
            
    def generate_response_instruction(self):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
    def generate_practice_instruction(self):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
            
    def respond(self, image_file, client_statement=None, max_tokens=128):
        if client_statement is not None:
            self._add_to_history("Client", client_statement)
        
        prompt = self.generate_response_instruction()
        prompt, image_tensor, image = self.build_prompt(prompt, image_file)
        response = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        proc_response = self.post_process_response(response)
        self._add_to_history("Therapist", proc_response)
        return proc_response
    
    
class LlavaMirrorEcCounselor(Counselor):
    def __init__(
        self, 
        args, 
    ):
        super().__init__(args=args)
        
    def generate_ec_instruction(self):
        return f"{IMAGE_DESC}\n{IMG_EMOTION_RECOG_DESC}"
            
    def generate_response_instruction(self, stage_direction):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Client Emotional State: {stage_direction}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Client Emotional State: {stage_direction}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
            
    def respond(self, image_file, client_statement=None, max_tokens=128):
        if client_statement is not None:
            self._add_to_history("Client", client_statement)
        ec_prompt = self.generate_ec_instruction()
        prompt, image_tensor, image = self.build_prompt(ec_prompt, image_file)
        stage_direction = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        
        # Build Prompt
        prompt = self.generate_response_instruction(stage_direction=stage_direction)
        prompt, image_tensor, image = self.build_prompt(prompt, image_file)
        response = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        proc_response = self.post_process_response(response)
        self._add_to_history("Therapist", proc_response)
        return proc_response


class LlavaMirrorPlanningCounselor(Counselor):
    def __init__(
        self, 
        args, 
    ):
        super().__init__(args=args)
        
    def generate_ec_instruction(self):
        return f"{IMAGE_DESC}\n{IMG_EMOTION_RECOG_DESC}"
    
    def parse_planning_value(self, planning_value):
        lines = planning_value.strip().split("\n")
        
        cbt_technique, cbt_plan_details = "", ""
        for line in lines:
            if line.startswith("- CBT technique:"):
                cbt_technique = line.replace("- CBT technique:", "").strip()
            elif line.startswith("- Counseling planning:"):
                cbt_plan_details = line.replace("- Counseling planning:", "").strip()

        cbt_plan = f"""- CBT technique: {cbt_technique}
- Counseling planning:
{cbt_plan_details}"""
        return cbt_plan
    
    def generate_planning_instruction(self):
        general_inst = f"""{IMAGE_DESC}
    You are a counselor specializing in CBT techniques. Your task is to use the provided client information, and dialogue to generate an appropriate CBT technique and a detailed counseling plan.

    Types of CBT Techniques:
    Efficiency Evaluation, Pie Chart Technique, Alternative Perspective, Decatastrophizing, Pros and Cons Analysis, Evidence-Based Questioning, Reality Testing, Continuum Technique, Changing Rules to Wishes, Behavior Experiment, Problem-Solving Skills Training, Systematic Exposure"""
        if len(self.history) == 0:
            return ''.join([
                f"{general_inst}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{PLANNING_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{general_inst}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{PLANNING_DESC}"
            ])
    
    def generate_response_instruction(self, cbt_plan):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{cbt_plan}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{cbt_plan}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
            
    def respond(self, image_file, client_statement=None, max_tokens=128):
        if client_statement is not None:
            self._add_to_history("Client", client_statement)

        planning_prompt = self.generate_planning_instruction()
        prompt, image_tensor, image = self.build_prompt(planning_prompt, image_file)
        if not self.cbt_plan:
            cbt_plan = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
            cbt_plan = self.parse_planning_value(planning_value=cbt_plan)
            self.cbt_plan = cbt_plan
        else:
            cbt_plan = self.cbt_plan
        
        prompt = self.generate_response_instruction(cbt_plan=cbt_plan)
        prompt, image_tensor, image = self.build_prompt(prompt, image_file)
        response = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        proc_response = self.post_process_response(response)
        self._add_to_history("Therapist", proc_response)
        return proc_response


class LlavaMirrorEcPlanningCounselor(LlavaMirrorPlanningCounselor):
    def __init__(
        self, 
        args, 
    ):
        super().__init__(args=args)
    
    def generate_response_instruction(self, stage_direction, cbt_plan):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Client Emotional State: {stage_direction}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{cbt_plan}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Client Emotional State: {stage_direction}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{cbt_plan}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
            
    def respond(self, image_file, client_statement=None, max_tokens=128):
        if client_statement is not None:
            self._add_to_history("Client", client_statement)
        ec_prompt = self.generate_ec_instruction()
        prompt, image_tensor, image = self.build_prompt(ec_prompt, image_file)
        stage_direction = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        
        planning_prompt = self.generate_planning_instruction()
        prompt, image_tensor, image = self.build_prompt(planning_prompt, image_file)
        if not self.cbt_plan:
            cbt_plan = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
            cbt_plan = self.parse_planning_value(planning_value=cbt_plan)
        else:
            cbt_plan = self.cbt_plan
        
        prompt = self.generate_response_instruction(stage_direction=stage_direction, cbt_plan=cbt_plan)
        prompt, image_tensor, image = self.build_prompt(prompt, image_file)
        response = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        proc_response = self.post_process_response(response)
        self._add_to_history("Therapist", proc_response)
        return proc_response


class LlavaCounselor(Counselor):
    def __init__(
        self, 
        args, 
    ):
        super().__init__(args=args)
            
    def generate_response_instruction(self):
        if len(self.history) == 0:
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"{CBT_DESC}"
            ])
        else:
            history = self._get_history_text(self.history, ctx_len=self.ctx_len)
            return ''.join([
                f"{IMAGE_DESC}\n",
                f"- Personal Information: {self.client_information}\n",
                f"- Reason for Counseling: {self.reason_counseling}\n",
                f"\nBelow is a conversation between the client and the psychotherapist.\n{history}\n\n",
                f"{CBT_DESC}"
            ])
            
    def respond(self, image_file, client_statement=None, max_tokens=128):
        if client_statement is not None:
            self._add_to_history("Client", client_statement)
        
        # Build Prompt
        prompt = self.generate_response_instruction()
        prompt, image_tensor, image = self.build_prompt(prompt, image_file)
        response = self.generate(prompt, image_tensor, image, max_tokens=max_tokens)
        proc_response = self.post_process_response(response)
        self._add_to_history("Therapist", proc_response)
        return proc_response
    