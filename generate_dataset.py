import os
import time
import random
import csv
import logging
import argparse
import io
import re
from sarvamai import SarvamAI

# Set up robust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_csv_from_text(text):
    """Extracts and parses CSV data directly from the AI's text response."""
    # 1. Clean markdown code blocks if the AI tries to be helpful
    if "```csv" in text:
        text = text.split("```csv")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    text = text.strip()
    if not text:
        return None

    try:
        # 2. Treat the raw string as a file to parse it safely
        csv_file = io.StringIO(text)
        reader = csv.DictReader(csv_file)
        
        # Lowercase headers in case the AI capitalized them (e.g., "Question" -> "question")
        if reader.fieldnames:
            reader.fieldnames = [str(field).strip().lower() for field in reader.fieldnames]
            
        # 3. Validate that we got the exact columns we need
        required_keys = ['question', 'answer', 'explanation']
        if not reader.fieldnames or not all(k in reader.fieldnames for k in required_keys):
            logger.error(f"Missing headers. Found: {reader.fieldnames}")
            return None

        # 4. Extract rows
        data = []
        for row in reader:
            # Ensure the row has data and isn't just empty columns
            if row.get('question') and row.get('answer'):
                data.append(row)
                
        return data
        
    except Exception as e:
        logger.error(f"Failed to parse CSV string. Error: {e}\nRaw content:\n{text[:200]}...")
        return None

def get_batch_with_retry(client, required_count, attempt=1, max_retries=5):
    """Fetches a batch with exponential backoff, requesting CSV format."""
    try:
        # Prompt explicitly enforces strict CSV format
        prompt = f"""Generate {required_count} unique, hyper-detailed science entries (Physics, Chemistry, Mathematics).
        Structure:
        1. Question: A deep, conceptual question based on first principles.
        2. Answer: A precise and accurate direct answer.
        3. Explanation: A hyper-detailed breakdown using low-level fundamental truths.

        Constraints:
        - Return ONLY raw CSV format.
        - The first row MUST be exactly this header: question,answer,explanation
        - You MUST enclose every field in double quotes (") so internal commas do not break the formatting.
        - Do NOT include any intro text, outro text, or markdown blocks. Just the CSV text.
        """

        response = client.chat.completions(
            model="sarvam-30b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
            stream=False
        )

        content = response.choices[0].message.content.strip()
        data = extract_csv_from_text(content)
        
        if data:
            return data
        raise ValueError("No valid CSV rows extracted.")

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
    parser = argparse.ArgumentParser(description="Generate Science Dataset via Sarvam AI (CSV Mode)")
    parser.add_argument("--total", type=int, default=100, help="Total rows to generate per run")
    parser.add_argument("--batch", type=int, default=5, help="Batch size per API call")
    parser.add_argument("--output", type=str, default="science_detailed_dataset.csv", help="Output CSV file name")
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

    logger.info(f"Starting CSV generation. Goal: {args.total} rows. Currently have: {generated_count}.")

    with open(args.output, "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        if not file_exists or os.path.getsize(args.output) == 0:
            writer.writeheader()

        while generated_count < args.total:
            batch_req = min(args.batch, args.total - generated_count)
            logger.info(f"Requesting batch of {batch_req}...")
            
            data = get_batch_with_retry(client, batch_req)
            
            if data and isinstance(data, list):
                valid_items = 0
                for item in data:
                    # Filter out any weird extra columns the AI might have added
                    filtered_item = {k: item.get(k, "").strip() for k in fieldnames}
                    
                    if all(filtered_item.values()):
                        writer.writerow(filtered_item)
                        valid_items += 1
                        generated_count += 1
                        
                        if generated_count >= args.total: 
                            break
                            
                csvfile.flush()
                logger.info(f"Successfully wrote {valid_items} items. Progress: {generated_count}/{args.total}")
            else:
                logger.error("Failed to retrieve valid batch. Waiting 5 seconds before next attempt.")
                time.sleep(5)

    logger.info(f"Generation complete. Dataset saved to {args.output}")

if __name__ == "__main__":
    main()
