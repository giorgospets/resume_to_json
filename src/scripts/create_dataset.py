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
        "providers": [
            "deepinfra/turbo",
            "crusoe/int8"
        ]
    },
    "qwen": {
        "temperature": 0.7,
        "top_p": 8,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "providers": [
            "chutes"
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


def extract_json_from_response(response: str) -> dict | None:
    """Extracts a JSON object from a model's string response."""
    try:
        # Remove ```json. ...``` tags
        if "```json" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from response: {e}\nResponse was: {response}")
        return None
        
# def are_all_values_extracted_from_text(response_json: dict, text: str, _id: str) -> bool:
#     """Returns True if all values are extracted from the text, False otherwise.
    
#     Values are strings, lists or dicts. Acts recursively for lists and dicts.
#     """
#     for key, value in response_json.items():
#         if isinstance(value, str):
#             if value not in " ".join([x.strip() for x in text.split()]):
#                 logger.warning(f"Item '{value}' for key '{key}' not found in text. ID: {_id}.")
#                 return False
#         elif isinstance(value, list):
#             for item in value:
#                 if isinstance(item, (str, int, float)):
#                     if str(item) not in " ".join([x.strip() for x in text.split()]):
#                         logger.warning(f"List item '{item}' for key '{key}' not found in text. ID: {_id}")
#                         return False
#                 elif isinstance(item, dict):
#                     if not are_all_values_extracted_from_text(item, text, _id):
#                         return False
#                 else:
#                     logger.warning(f"Unsupported list item type for key '{key}': {type(item)}, ID: {_id}")
#                     return False
#         elif isinstance(value, dict):
#             if not are_all_values_extracted_from_text(value, text, _id):
#                 return False
#         else:
#             logger.warning(f"Unsupported value type for key '{key}': {type(value)}")
#             return False
#     return True

def create_conversation(cv_text: str) -> List[Dict[str, Any]]:
    few_shot_examples = [
        (EXAMPLE_1, RESPONSE_1),
        (EXAMPLE_2, RESPONSE_2)
    ]
    conversation_base = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in few_shot_examples:
        conversation_base.extend([
            {"role": "user", "content": item[0]},
            {"role": "assistant", "content": json.dumps(item[1], ensure_ascii=False)}
        ])

    return conversation_base + [{
        "role": "user",
        "content": cv_text
    }]

def fill_dataset(
    dataset_entries_list: List[dict],
    model_name: str,
    rpm_limit: int | None = None,
    return_every: int = 50
) -> Iterator[List[Dict[str, Any]]]:
    """Extracts metadata by sending requests in parallel with improved error handling."""
    model_key = next((key for key in CONFIG if key in model_name), "llama")
    config = CONFIG[model_key]
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    def send_request(row_dict: dict) -> dict:
        ret_dict = row_dict.copy()

        # Apply rate limiting before making the request
        if rpm_limit:
            rate_limiter.wait_if_needed(model_name, rpm_limit, min_interval=0.05)

        conversation = create_conversation(row_dict["Text"])
        
        max_retries = 2
        base_delay = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    extra_body={
                        "usage": {"include": True},
                        **({
                            "provider": {
                                "only": config["providers"]
                            }
                        } if "providers" in config else {})
                    }
                )
                response_json = extract_json_from_response(response.choices[0].message.content)
                if not response_json:
                    continue

                # if not are_all_values_extracted_from_text(response_json, row_dict["Text"], row_dict["ID"]):
                #     print(response_json)
                #     print("****************************")
                #     conversation += [{
                #         "role": "user",
                #         "content": (
                #             "Some value-strings are not extracted from the CV. "
                #             "Respond with a JSON where all non-empty strings are "
                #             "*extracted* from the CV. Do not provide any "
                #             "additional information."
                #         )
                #     }]
                #     continue  

                ret_dict["timestamp"] = pendulum.now("Europe/Athens").strftime("%Y-%m-%d %H:%M:%S")
                
                ret_dict["json"] = response_json
                return ret_dict
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "404" in error_str and attempt < max_retries - 1:
                    delay = min(300, base_delay * (2 ** attempt))
                    logger.warning(
                        f"Provider unavailable (404) for {row_dict['ID']}, retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                    
                elif ("429" in error_str or "rate" in error_str) and attempt < max_retries - 1:
                    # Rate limiting - shorter delays
                    delay = min(120, base_delay * (1.5 ** attempt))
                    logger.warning(
                        f"Rate limit hit for {ret_dict['ID']}, retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                    
                elif ("503" in error_str or "502" in error_str) and attempt < max_retries - 1:
                    # Server errors - medium delays
                    delay = min(180, base_delay * (1.8 ** attempt))
                    logger.warning(
                        f"Server error for {ret_dict['ID']}, retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Error for {ret_dict['ID']}: {e}")
                    break

        ret_dict["json"] = None
        return ret_dict

    max_workers = 16

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cv = {
            executor.submit(send_request, row_dict): row_dict
            for row_dict in dataset_entries_list
        }
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_cv):
            try:
                result = future.result(timeout=60)
                if result:
                    results.append(result)
                    if not result["json"]:
                        logger.warning(f"Failed: {result['ID']}")
            except concurrent.futures.TimeoutError:
                row_dict = future_to_cv[future]
                logger.error(f"Timeout processing {row_dict['ID']}")
                results.append({"ID": row_dict["ID"], "json": None})
            except Exception as e:
                row_dict = future_to_cv[future]
                logger.error(f"Failed processing {row_dict['ID']}: {e}")
                results.append({"ID": row_dict["ID"], "json": None})

            if len(results) >= return_every:
                yield results
                results = []

    if results:
        yield [r for r in results if r["json"] is not None]


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from headers using LLM.")
    parser.add_argument("--model_index", type=int, default=2, help="Index of the model to use from the MODELS list.")
    parser.add_argument("--number_limit", type=int, default=100000, help="Number of headers to process in this run.")
    parser.add_argument("--rpm_limit", type=int, default=1000)
    args = parser.parse_args()

    MODELS = [
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.5-flash-lite",
        "qwen/qwen3-235b-a22b-2507",
    ]

    model_name = MODELS[args.model_index]
    logger.info(f"Using model: {model_name}")

    with open(
        os.path.join(PROJECT_ROOT, "data/preprocessed_dataset.json"), 
        "rb"
    ) as f:
        input: list[dict] = json.load(f)

    output_filepath = os.path.join(PROJECT_ROOT, "data/structured_dataset.json")

    try:
        with open(output_filepath) as f:
            content = f.read()
            if content.strip():
                existing_dataset: list[dict] = json.loads(content)
            else:
                existing_dataset = []
    except FileNotFoundError:
        existing_dataset = []

    # Only process unprocessed IDS 
    processed = {x["ID"] for x in existing_dataset}
    entries_to_process = [
        x
        for x in input
        if x["ID"] not in processed
    ]

    if not entries_to_process:
        logger.info("No more entries to process")
    
    # Sort by ID
    entries_to_process = sorted(entries_to_process, key=lambda x: x["ID"])

    # Limit the number of entries to process in this run
    entries_to_process = entries_to_process[:args.number_limit]
    logger.info(f"Processing {len(entries_to_process)} new entries.")

    logger.info(f"Using rpm: {args.rpm_limit}")
        
    for new_filled_entries in fill_dataset(
        dataset_entries_list=entries_to_process,
        model_name=model_name,
        rpm_limit=args.rpm_limit,
        return_every=50
    ):
        if not new_filled_entries:
            logger.info("No new filled entries")
            continue

        # Append new results to existing data
        existing_dataset.extend(new_filled_entries)
        # Save after each batch
        with open(output_filepath, "w") as f:
            json.dump(existing_dataset, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Total processed: {len(existing_dataset)}/"
            f"{len(entries_to_process) + len(existing_dataset)}"
        )

    logger.info("Finished processing all batches")


if __name__ == "__main__":
    main()
    