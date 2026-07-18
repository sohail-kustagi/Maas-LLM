import os
import json
import asyncio
import urllib.request
from llama_cpp import Llama

try:
    from core.validation import CommandValidationError, validate_commander_output
except ImportError:
    from src.core.validation import CommandValidationError, validate_commander_output

class CommanderNode:
    def __init__(self, model_repo="microsoft/Phi-3-mini-4k-instruct-gguf", model_file="Phi-3-mini-4k-instruct-q4.gguf"):
        self.model_path = os.path.join(os.getcwd(), model_file)
        self.download_model(model_repo, model_file)
        
        print(f"[Commander] Loading LLM from {self.model_path}...")
        # Since this runs on an IdeaPad with 16GB RAM and Core Ultra, we will use CPU/OpenBLAS natively
        # n_ctx limits context to save RAM. n_threads automatically scales to CPU cores.
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=8, 
            verbose=False
        )
        print("[Commander] LLM Loaded successfully.")

    def download_model(self, model_repo, model_file):
        """Downloads a small GGUF model if it doesn't already exist."""
        if not os.path.exists(self.model_path):
            print(f"[Commander] Model {model_file} not found locally. Downloading from HuggingFace...")
            url = f"https://huggingface.co/{model_repo}/resolve/main/{model_file}"
            
            def report(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                if percent % 10 == 0:
                    print(f"\rDownloading: {percent}%", end="")
            
            urllib.request.urlretrieve(url, self.model_path, reporthook=report)
            print("\n[Commander] Download complete.")

    async def generate_mavlink_command(self, context_prompt: str, telemetry=None):
        print("\n[Commander] Triggered! Generating MAVLink routing command...")
        
        # We enforce a strict JSON output representing a MAVLink SET_POSITION_TARGET_LOCAL_NED command
        system_prompt = (
            "You are an autonomous drone fleet commander. You receive context about a disaster anomaly. "
            "You must output a strictly formatted JSON object representing a MAVLink navigation command to reroute the drone to investigate."
            "\n\nJSON SCHEMA:\n"
            "{\n"
            '  "command": "SET_POSITION_TARGET_LOCAL_NED",\n'
            '  "target_system": 1,\n'
            '  "target_component": 1,\n'
            '  "x": <float, offset in meters North>,\n'
            '  "y": <float, offset in meters East>,\n'
            '  "z": <float, offset in meters Down (negative for altitude)>,\n'
            '  "reasoning": "<short string explaining why you chose this coordinate based on the context>"\n'
            "}"
        )
        
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{context_prompt}<|end|>\n<|assistant|>\n"
        
        # In a real production setup, we would compile a LlamaGrammar for strict JSON syntax enforcement.
        # For simplicity in this sandbox, we prompt heavily for JSON.
        
        # Run inference in a thread to not block the asyncio event loop
        loop = asyncio.get_event_loop()
        
        def run_inference():
            return self.llm(
                prompt,
                max_tokens=256,
                stop=["<|end|>"],
                temperature=0.1
            )
            
        response = await loop.run_in_executor(None, run_inference)
        
        output_text = response['choices'][0]['text'].strip()
        print("[Commander] Generated Command:")
        print(output_text)
        
        try:
            command_json = json.loads(output_text)
            validated_command = validate_commander_output(command_json, telemetry)
            return validated_command.as_dict()
        except (json.JSONDecodeError, CommandValidationError) as error:
            print(f"[Commander] ERROR: Invalid command output: {error}")
            return None
