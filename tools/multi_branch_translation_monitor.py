import os
import hashlib
import subprocess
import json
import logging
import time
import argparse

# Set up logging
logging.basicConfig(filename='translation_monitor.log', level=logging.INFO)

def md5(file_path):
    """Computes the MD5 hash of the given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b"")):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def execute_font_generation_script(retry_count=3):
    """Executes the font generation script with a retry mechanism."""
    for attempt in range(retry_count):
        try:
            subprocess.run(['python', 'selfdrive/assets/fonts/process.py'], check=True)
            logging.info('Font generation script executed successfully.')
            return
        except subprocess.CalledProcessError as e:
            logging.error(f'Error executing font generation script: {e}')
            time.sleep(2)  # wait before retry
    logging.error('Font generation script failed after retries.')

def monitor_translation_files(repo_root, branches, interval, once):
    """Monitors translation files for changes."""
    hashes = {}
    while True:
        for branch in branches:
            file_path = os.path.join(repo_root, branch, 'translations', 'messages.po')
            current_hash = md5(file_path)

            if branch not in hashes:
                hashes[branch] = current_hash
                continue

            if current_hash != hashes[branch]:
                logging.info(f'File changed detected in {branch}.')
                execute_font_generation_script()  # Execute on change
                hashes[branch] = current_hash

        if once:
            break
        time.sleep(interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor translation files.')
    parser.add_argument('--repo-root', required=True, help='The root directory of the repository.')
    parser.add_argument('--branches', nargs='+', required=True, help='Branches to monitor.')
    parser.add_argument('--interval', type=int, default=60, help='Interval to check for changes (in seconds).')
    parser.add_argument('--once', action='store_true', help='Run the monitor once and exit.')
    args = parser.parse_args()

    monitor_translation_files(args.repo_root, args.branches, args.interval, args.once)
