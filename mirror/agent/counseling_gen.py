from mirror.agent.base import Agent
from pathlib import Path
from os.path import join as pjoin
current_dir  = Path(__file__)

QUERY = """## Client Information ##

### Personal Information ###: 
{client_persona}

### Personality Traits ###: {personality_trait}
### Distorted Thoughts ###: {distorted_thoughts}
### Thinking Trap ###: {cognitive_distortion}
### Reason for Seeking Counseling ###: {reason_counseling}

## CBT Plan ##
{cbt_plan}

**KEEP ALL RESPONSE TO MAXIMUM OF 2 LINES.**
## First Session Counseling ##"""


class CounselingGenerator(Agent):
    def __init__(self, model="gpt-4o", version="default"):
        super().__init__(model=model)
        self.system_message = self.load_file(
            pjoin(current_dir.parent, f"prompts/session_{version}.txt")
        )
        self._load_fewshot(pjoin(current_dir.parent, "examples"))
        self.model = model

    def _load_fewshot(self, sample_dirpath):
        from glob import iglob
        self.fewshot = []
        for prompt_path in iglob(pjoin(sample_dirpath, "prompt_*.txt")):
            suffix = prompt_path.split("/")[-1].split("_")[-1]
            self.fewshot += [{
                "role": "user",
                "content": self.load_file(prompt_path)
            }, {
                "role": "assistant",
                "content": self.load_file(
                    pjoin(sample_dirpath, f"output_{suffix}")
                )
            }]

    def _build_message(self, client_persona, distorted_thoughts, cognitive_distortion, personality_trait, cbt_plan, reason_counseling):
        messages = [{"role": "system", "content": self.system_message}]
        messages += self.fewshot
        messages += [{
                'role': "user",
                'content': QUERY.format(
                    client_persona=client_persona,
                    distorted_thoughts=distorted_thoughts,
                    cognitive_distortion=cognitive_distortion,
                    personality_trait=personality_trait,
                    cbt_plan=cbt_plan,
                    reason_counseling=reason_counseling
                )   
            }]
        return messages
    
    def get_response(self, client_persona, distorted_thoughts, cognitive_distortion, personality_trait, cbt_plan, reason_counseling):
        messages = self._build_message(client_persona,
                                        distorted_thoughts, 
                                        cognitive_distortion,
                                        personality_trait, cbt_plan, reason_counseling)
        response = self.request(messages)
        return response
    
    def get_message(self, custom_id, client_persona, distorted_thoughts, cognitive_distortion, personality_trait, cbt_plan, reason_counseling):
        messages = self._build_message(client_persona,
                                        distorted_thoughts, 
                                        cognitive_distortion,
                                        personality_trait, cbt_plan, reason_counseling)
        
        return {
            "custom_id": f"{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model, 
                "messages": messages,
                "max_tokens": 1024, 
                "temperature": 0.7,      
                "top_p": 0.9,             
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
            },
        }
    
