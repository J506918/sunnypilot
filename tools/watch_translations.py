import os
import time
import logging
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Change these variables as needed
TRANSLATIONS_DIR = 'selfdrive/ui/translations'
FONT_PROCESS_SCRIPT = 'selfdrive/assets/fonts/process.py'
RETRY_COUNT = 5
SLEEP_TIME = 5  # Seconds between retries

class TranslationChangeHandler(FileSystemEventHandler):
    """Handles events when translation files change."""
    def on_modified(self, event):
        if event.src_path.endswith('.json'):  # Assuming translation files are JSON
            logging.info(f"Detected change in {event.src_path}. Triggering font generation...")
            self.trigger_font_generation()

    def trigger_font_generation(self):
        for attempt in range(RETRY_COUNT):
            try:
                logging.info("Starting font generation script...")
                subprocess.run(['python', FONT_PROCESS_SCRIPT], check=True)
                logging.info("Font generation completed successfully.")
                break
            except subprocess.CalledProcessError:
                logging.error(f"Font generation failed (attempt {attempt + 1}). Retrying...")
                time.sleep(SLEEP_TIME)
        else:
            logging.error("Font generation failed after maximum retries.")

def monitor_translations(continuous=False):
    """Monitors the translations directory for changes."""
    event_handler = TranslationChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, TRANSLATIONS_DIR, recursive=False)
    observer.start()
    logging.info("Started monitoring translations directory.")
    
    try:
        if continuous:
            while True:
                time.sleep(1)  # Keep script running
        else:
            time.sleep(10)  # Monitor for a short period and then exit
    finally:
        observer.stop()
        observer.join()
        logging.info("Stopped monitoring translations directory.")

if __name__ == "__main__":
    # Run in continuous monitoring mode or one-time execution
    monitor_translations(continuous=True)  # Change to False for one-time execution
