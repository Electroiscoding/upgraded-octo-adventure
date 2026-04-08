import os
import json
import time
import random
import csv
import logging
import argparse
import re
from sarvamai import SarvamAI

# Set up robust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_text(text):
    """Bulletproof JSON extraction using regex to handle LLM formatting quirks."""
    try:
        # Look for JSON arrays specifically
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # Fallback to general JSON loading
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON. Raw content: {text[:200]}...")
        return None

def get_batch_with_retry(client, required_count, attempt=1, max_retries=5):
    """Fetches a batch of detailed science Q&A pairs with exponential backoff."""
    try:
        prompt = f"""Generate {required_count} unique, hyper-detailed science entries (Physics, Chemistry, Mathematics).
        Structure:
        1. Question: A deep, conceptual question based on first principles.
        2. Answer: A precise and accurate direct answer.
        3. Explanation: A hyper-detailed breakdown using low-level fundamental truths.

        Constraints:
        - Return ONLY a valid JSON list of objects with exactly these keys: 'question', 'answer', 'explanation'.
        - Do not include any other text before or after the JSON.
        """

        response = client.chat.completions(
            model="sarvam-30b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
            stream=False
        )

        content = response.choices[0].message.content.strip()
        data = extract_json_from_text(content)
        
        if data:
            return data
        raise ValueError("No valid JSON found in response.")

    except Exception as e:
        logger.warning(f"Attempt {attempt} failed: {e}")
        if attempt <= max_retries:
            wait_time = (2 ** attempt) + random.random()
            logger.info(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            return get_batch_with_retry(client, required_count, attempt + 1, max_retries)
        logger.error("Max retries reached. Returning empty batch.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate Science Dataset via Sarvam AI")
    parser.add_argument("--total", type=int, default=100, help="Total rows to generate per run")
    parser.add_argument("--batch", type=int, default=5, help="Batch size per API call")
    parser.add_argument("--output", type=str, default="science_dataset.csv", help="Output CSV file name")
    args = parser.parse_args()

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        logger.error("SARVAM_API_KEY environment variable is missing!")
        exit(1)

    client = SarvamAI(api_subscription_key=api_key)
    fieldnames = ['question', 'answer', 'explanation']
    
    # Calculate how many we already have to resume properly
    generated_count = 0
    file_exists = os.path.exists(args.output)
    if file_exists:
        with open(args.output, 'r', encoding="utf-8") as f:
            generated_count = max(0, sum(1 for row in f) - 1) # Subtract header

    logger.info(f"Starting generation. Goal: {args.total} rows. Currently have: {generated_count}.")

    with open(args.output, "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(args.output) == 0:
            writer.writeheader()

        while generated_count < args.total:
            batch_req = min(args.batch, args.total - generated_count)
            logger.info(f"Requesting batch of {batch_req}...")
            
            data = get_batch_with_retry(client, batch_req)
            
            if data and isinstance(data, list):
                valid_items = 0
                for item in data:
                    filtered_item = {k: item.get(k, "") for k in fieldnames}
                    # Ensure all fields have some content
                    if all(filtered_item.values()):
                        writer.writerow(filtered_item)
                        valid_items += 1
                        generated_count += 1
                        
                        if generated_count >= args.total: 
                            break
                            
                csvfile.flush() # Force write to disk immediately
                logger.info(f"Successfully wrote {valid_items} items. Progress: {generated_count}/{args.total}")
            else:
                logger.error("Failed to retrieve valid batch. Waiting before next attempt.")
                time.sleep(5)

    logger.info(f"Generation complete. Dataset saved to {args.output}")

if __name__ == "__main__":
    main()
