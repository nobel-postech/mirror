import os
import json
from pathlib import Path

from typing import List, Union
from os.path import join as pjoin
from parlai.core.message import Message
from parlai.core.agents import create_agent

class Canary(object):
    def __init__(self, canary_dir):
        canary_meta_data = pjoin(canary_dir, 'model.opt')
        with open(canary_meta_data) as f:
            opt = json.load(f)

        opt['skip_generation'] = False
        opt['model_file'] = pjoin(canary_dir, 'model')
        self.agent = create_agent(opt)

    def chirp(self, input: Union[str, List]):
        if isinstance(input, str):
            input = [input]

        return self.get_batch_output(input)
    
    def get_output(self, input: str):
        return self.agent.respond(Message(text=input))

    def get_batch_output(self, batch_input: List[str]):
        message_batch = []
        for input in batch_input:
            message_batch.append(Message(text=input))

        return self.agent.batch_respond(message_batch)
    
    def reset(self):
        self.agent.reset()