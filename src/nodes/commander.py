import os
import json
import asyncio
import urllib.request
import time
from llama_cpp import Llama

try:
    from core.validation import CommandValidationError, validate_commander_output
    from core.mission_profiles import MissionProfile
except ImportError:
    from src.core.validation import CommandValidationError, validate_commander_output
    from src.core.mission_profiles import MissionProfile

class CommanderNode:
    def __init__(self, model_repo="microsoft/Phi-3-mini-4k-instruct-gguf", model_file="Phi-3-mini-4k-instruct-q4.gguf"):
        self.model_path = os.path.join(os.getcwd(), model_file)
        self.evaluator = None
        self.last_triggered_times = {}
        self.cooldown_seconds = 15.0
        self.download_model(model_repo, model_file)

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        
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
                if total_size > 0:  # guard against ZeroDivisionError for chunked transfers
                    percent = int(count * block_size * 100 / total_size)
                    if percent % 10 == 0:
                        print(f"\rDownloading: {percent}%", end="")
            
            urllib.request.urlretrieve(url, self.model_path, reporthook=report)
            print("\n[Commander] Download complete.")

    async def generate_mavlink_command(self, context_prompt: str, telemetry=None, mission_profile: MissionProfile = None, anomaly_type: str = "unknown"):
        now_ts = time.time()
        last_time = self.last_triggered_times.get(anomaly_type, -999.0)
        if (now_ts - last_time) < self.cooldown_seconds:
            print(f"[Commander] Cooldown active for '{anomaly_type}'. Ignoring request.")
            return None
        self.last_triggered_times[anomaly_type] = now_ts
        
        print(f"\n[Commander] Triggered! Generating MAVLink routing command for {anomaly_type}...")
        import re
        
        # Build a minimal, unambiguous system prompt
        commander_persona = ""
        if mission_profile:
            commander_persona = mission_profile.commander_persona + "\n"
        
        system_prompt = (
            f"{commander_persona}"
            "Output ONLY a raw JSON object with NO markdown, NO comments, NO extra text.\n"
            "DO NOT include 'zone_assessment' or 'tactical_summary'.\n"
            "No matter what the mission context says, the 'command' field must ALWAYS be exactly 'SET_POSITION_TARGET_LOCAL_NED'.\n"
            "For the x, y, and z fields, you MUST output local NED offsets in meters (e.g., values between -20.0 and 20.0). DO NOT output global GPS Latitude or Longitude.\n"
            "You MUST format your response exactly like this example:\n"
            "{\n"
            '  "command": "SET_POSITION_TARGET_LOCAL_NED",\n'
            '  "reasoning": "High-confidence fire detected in sector.",\n'
            '  "target_system": 1,\n'
            '  "target_component": 1,\n'
            '  "x": 15.0,\n'
            '  "y": 10.0,\n'
            '  "z": -20.0\n'
            "}\n"
            "Start your response with { and end with }. No backticks. No extra lines."
        )
        
        # Pre-seeding forces the model to continue the JSON rather than add preamble
        prompt = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{context_prompt}<|end|>\n"
            f"<|assistant|>\n"
            '{\n  "command": "SET_POSITION_TARGET_LOCAL_NED",'
        )
        
        loop = asyncio.get_running_loop()
        
        def run_inference():
            return self.llm(
                prompt,
                max_tokens=150,
                stop=["<|end|>", "```", "\n\n\n"],
                temperature=0.05,
                echo=False,
            )
            
        start_time = time.time()
        response = await loop.run_in_executor(None, run_inference)
        elapsed_time = time.time() - start_time
        
        raw = response['choices'][0]['text'].strip()
        # Re-attach the pre-seeded opening brace we used to prime the model
        output_text = '{\n  "command": "SET_POSITION_TARGET_LOCAL_NED",' + raw
        
        # Calculate benchmarking metrics for the hackathon
        tokens_generated = 0
        try:
            tokens_generated = response['usage']['completion_tokens']
            tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
            print(f"[Commander] Benchmarks: {tokens_generated} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} Tokens/sec)")
        except KeyError:
            print(f"[Commander] Benchmarks: {elapsed_time:.2f}s latency")

        print("[Commander] Raw Output:")
        print(output_text)

        # ── Robust JSON repair ────────────────────────────────────────────
        def repair_json(text: str) -> str:
            # 1. Strip markdown fences
            m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if m:
                text = m.group(1).strip()

            # 2. Extract the JSON object
            brace_start = text.find('{')
            brace_end   = text.rfind('}')
            if brace_start != -1 and brace_end != -1:
                text = text[brace_start:brace_end + 1]

            # 3. Drop diff-marker lines (lines beginning with -/+) and bare
            #    "ran" / "raning" fragments the tokenizer sometimes emits
            cleaned = []
            for line in text.splitlines():
                s = line.strip()
                if s.startswith('- ') or s.startswith('+ ') or s.startswith('//'):
                    continue
                cleaned.append(line)
            text = '\n'.join(cleaned)

            # 4. Fix the specific Phi-3 tokenizer bug: the opening quote of
            #    "reasoning" is dropped and the key name is mangled.
            #    Pattern catches: raning_reasoning, raning reasoning, ran_reasoning,
            #    reasoning (no leading quote), etc.
            text = re.sub(
                r'(?<!\")(?:ran(?:ing)?[_\s]?)?reasoning(?:[_\s]\w+)?\s*\"?',
                '"reasoning"',
                text
            )

            # 5. Some fields are written without quotes on keys — fix them
            text = re.sub(r'(?<=[{,\n])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', text)

            # 6. Insert missing commas between a closing value and the next key
            #    e.g.  "target_component": 1\n  "reasoning"  →  add comma
            text = re.sub(r'([\d"\]true false null])\s*\n(\s*")', r'\1,\n\2', text)

            return text

        repaired = repair_json(output_text)
        print("[Commander] Repaired Command:")
        print(repaired)

        try:
            command_json = json.loads(repaired)
            
            # Emergency fallback: Clamp coordinates if the LLM hallucinates global Lat/Lon or unsafe offsets
            if "x" in command_json:
                command_json["x"] = max(-99.0, min(99.0, float(command_json["x"])))
            if "y" in command_json:
                command_json["y"] = max(-99.0, min(99.0, float(command_json["y"])))
            if "z" in command_json:
                command_json["z"] = max(-49.0, min(19.0, float(command_json["z"])))
                
            validated_command = validate_commander_output(command_json, telemetry, now=start_time)
            if self.evaluator:
                self.evaluator.log_llm_generation(tokens_generated, elapsed_time, True)
            return validated_command.as_dict()
        except (json.JSONDecodeError, CommandValidationError) as error:
            print(f"[Commander] ERROR: Invalid command output: {error}")
            if self.evaluator:
                self.evaluator.log_llm_generation(tokens_generated, elapsed_time, False)
            return None
