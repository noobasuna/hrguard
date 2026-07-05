from openai import OpenAI
from utils import config
import os
import random
import time


class Attack:
    def __init__(self):
        super()

    def _config_value(self, env_name, key_name, keys_name=None):
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
        if hasattr(config, key_name):
            return getattr(config, key_name)
        if keys_name and hasattr(config, keys_name):
            values = [value for value in getattr(config, keys_name) if value and value != "your tokens"]
            if values:
                return random.choice(values)
        return None

    def init_model(self, model):
        model_lower = model.lower()
        if model_lower.startswith("deepseek"):
            print("Processing with DeepSeek API")
            api_key = self._config_value("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "DEEPSEEK_KEYS")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is not set and DEEPSEEK_KEY(S) not found in config")
            base_url = (
                os.getenv("DEEPSEEK_BASE_URL")
                or getattr(config, "DEEPSEEK_URL", None)
                or "https://api.deepseek.com"
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
        elif "gpt-4o" in model or model_lower.startswith("gpt-"):
            print("Processing with OPENAI API")
            api_key = self._config_value("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_KEYS")
            if not api_key:
                raise ValueError("Neither OPENAI_KEY nor OPENAI_KEYS found in config")
            client = OpenAI(api_key=api_key)
        else:
            print("Processing with OHMYGPT")
            api_key = self._config_value("OHMYGPT_API_KEY", "OHMYGPT_KEY", "OHMYGPT_KEYS")
            if not api_key:
                raise ValueError("Neither OHMYGPT_KEY nor OHMYGPT_KEYS found in config")
            if not hasattr(config, "OHMYGPT_URL"):
                raise ValueError("OHMYGPT_URL not found in config")
            client = OpenAI(
                api_key=api_key, base_url=config.OHMYGPT_URL
            )
        return client
        """   
        else:
            print("Processing with ZHIZZ")
            if hasattr(config, 'ZHI_KEY'):
                api_key = config.ZHI_KEY
            elif hasattr(config, 'ZHI_KEYS'):
                api_key = random.choice(config.ZHI_KEYS)
            else:
                raise ValueError("Neither ZHI_KEY nor ZHI_KEYS found in config")
            client = OpenAI(
                api_key=api_key, base_url=config.ZHI_URL
            )
        """  

    def get_response(self, messages, client, model):
        retry_count = 0
        while retry_count < 5:
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages
                )
                return response
            except Exception as e:
                retry_count += 1
                print(f"Error occurred: {e}. Retrying {retry_count}/5...")
                time.sleep(2) 

        print("Max retries reached. Returning empty response.")
        return {}
