"""Script to fill fill the "json" column of the dataset by LLM responses."""

import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import sys
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import argparse
import logging
import time
import json
import concurrent.futures
from typing import List, Dict, Any, Iterator
import pendulum
from openai import OpenAI

import threading
from collections import defaultdict


from utils.dataset_creation_prompts import (
    SYSTEM_PROMPT,
    EXAMPLE_1, RESPONSE_1,
    EXAMPLE_2, RESPONSE_2
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Configuration for different models
CONFIG = {
    "gemma": {
        "temperature": 1.0,
        "top_p": 0.95,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "llama": {
        "temperature": 0.7,
        "top_p": 0.95,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "qwen": {
        "temperature": 0.7,
        "top_p": 8,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "providers": [
            "targon/bf16", 
            # "deepinfra/fp8"
        ]
    },
    "DeepSeek": {
        "temperature": 0.3,
        "top_p": 0.95,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    },
    "gemini": {
        "temperature": 1.0,
        "top_p": 0.95,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "providers": ["google-vertex"]
    },
}


class RateLimiter:
    """Rate limiter that prevents rate limit violations."""
    
    def __init__(self):
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._request_times: Dict[str, list] = defaultdict(list)
        self._last_request_time: Dict[str, float] = defaultdict(float)
    
    def wait_if_needed(self, model_name: str, rpm_limit: int, min_interval: float = 1.0) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            model_name: The model identifier
            rpm_limit: Requests per minute limit
            min_interval: Minimum seconds between requests (default 1.0)
        """
        with self._locks[model_name]:
            now = time.time()
            
            # Clean old requests (older than 1 minute)
            self._request_times[model_name] = [
                req_time for req_time in self._request_times[model_name]
                if now - req_time < 60
            ]
            
            # Check if we need to wait based on RPM limit
            if len(self._request_times[model_name]) >= rpm_limit:
                # Wait until the oldest request is more than 60 seconds old
                oldest_request = self._request_times[model_name][0]
                wait_time = 60 - (now - oldest_request) + 0.1
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
            
            # Check minimum interval between requests
            last_request = self._last_request_time[model_name]
            if last_request > 0:
                time_since_last = now - last_request
                if time_since_last < min_interval:
                    wait_time = min_interval - time_since_last
                    time.sleep(wait_time)
                    now = time.time()
            
            # Record this request
            self._request_times[model_name].append(now)
            self._last_request_time[model_name] = now
    
    def reset(self, model_name: str | None = None) -> None:
        """Reset rate limiting data for a model or all models."""
        if model_name:
            with self._locks[model_name]:
                self._request_times[model_name].clear()
                self._last_request_time[model_name] = 0
        else:
            for model in list(self._locks.keys()):
                self.reset(model)

rate_limiter = RateLimiter()

def extract_json_from_response(response: str) -> dict | json.JSONDecodeError:

    """Extracts a JSON object from a model's string response."""
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from response: {e}\nResponse was: {response}")
        raise e

def create_request(
    id: str,
    content: str,
    model_name: str,
    system_prompt: str,
    few_shot_examples: List[tuple]
) -> List[Dict[str, Any]]:
    """Prepares a list of requests for the batch API with caching enabled."""
    try:
        conversation_base = [{"role": "system", "content": system_prompt}]
        for item in few_shot_examples:
            conversation_base.extend([
                {"role": "user", "content": item[0]},
                {"role": "assistant", "content": json.dumps(item[1], ensure_ascii=False)}
            ])

        return {
            "custom_id": id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": conversation_base.append({
                    "role": "user",
                    "content": content
                }),
                "provider": {
                    "data_collection": "allow",
                    "allow_fallbacks": False 
                },
                "stream": False, 
            }
        }
    except Exception as e:
        logger.error(f"The following error occurred while preparing the requests: {e}")
        return None

def send_request(
    id: str,
    model_name: str,
    config: dict,
    client: OpenAI,
    messages: List[Dict[str, str]],
    rpm_limit: int | None = None,
) -> OpenAI.ChatCompletion | None:

    # Apply rate limiting before making the request
    if rpm_limit:
        rate_limiter.wait_if_needed(model_name, rpm_limit, min_interval=1.0)
    
    max_retries = 5
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={
                    "provider": {
                        "only": config["providers"]
                    }
                } if "providers" in config else {}
            )
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "404" in error_str and attempt < max_retries - 1:
                delay = min(300, base_delay * (2 ** attempt))
                logger.warning(
                    f"Provider unavailable (404) for {id}, retrying in "
                    f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
                
            elif ("429" in error_str or "rate" in error_str) and attempt < max_retries - 1:
                # Rate limiting - shorter delays
                delay = min(120, base_delay * (1.5 ** attempt))
                logger.warning(
                    f"Rate limit hit for {id}, retrying in "
                    f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
                
            elif ("503" in error_str or "502" in error_str) and attempt < max_retries - 1:
                # Server errors - medium delays
                delay = min(180, base_delay * (1.8 ** attempt))
                logger.warning(
                    f"Server error for {id}, retrying in "
                    f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
            else:
                logger.error(f"Error for {id}: {e}")
                break

    return None

def get_base_conversation(
    system_prompt: str,
    few_shot_examples: List[tuple]
) -> List[Dict[str, str]]:
    """Constructs the base conversation with system prompt and few-shot examples."""
    conversation = [{"role": "system", "content": system_prompt}]
    for item in few_shot_examples:
        conversation.extend([
            {"role": "user", "content": item[0]},
            {"role": "assistant", "content": json.dumps(item[1], ensure_ascii=False)}
        ])
    return conversation

# TODO: Finish this
def fill_dataset(
    system_prompt: str,
    few_shot_examples: List[tuple],
    dataset_entries_list: List[dict],
    column_name: str,
    model_name: str,
    rpm_limit: int | None = None,
    return_every: int = 50
) -> Iterator[List[Dict[str, Any]]]:
    """Extracts metadata by sending requests in parallel with improved error handling."""
    model_key = next((key for key in CONFIG if key in model_name), "llama")
    config = CONFIG[model_key]
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    max_workers = 16

    base_conversation = get_base_conversation(
        system_prompt,
        few_shot_examples
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cv = {
            executor.submit(
                send_request, 
                cv_dict["ID"], 
                model_name,
                config,
                client,
                base_conversation + {"role": "user", "content": cv_dict[column_name]},
                rpm_limit
            ): cv_dict
            for cv_dict in dataset_entries_list
        }
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_cv):
            try:
                result = future.result(timeout=60)
                if result:
                    results.append(result)
            except concurrent.futures.TimeoutError:
                cv_dict = future_to_cv[future]
                logger.error(f"Timeout processing {cv_dict['ID']}")
            except Exception as e:
                cv_dict = future_to_cv[future]
                logger.error(f"Failed processing {cv_dict['ID']}: {e}")

            if len(results) >= return_every:
                yield results
                results = []

    if results:
        yield [r for r in results if r["json"] is not None]