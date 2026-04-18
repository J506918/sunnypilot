import os
import time
import logging
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Setup logging
logging.basicConfig(filename='translation_monitor.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.po'):
            logging.info(f'File changed: {event.src_path}')
            self.run_font_generation()

    def run_font_generation(self):
        try:
            subprocess.run(['python3', 'selfdrive/assets/fonts/process.py'], check=True)
            logging.info('Font generation executed successfully.')
        except subprocess.CalledProcessError as e:
            logging.error(f'Error during font generation: {e}')

def start_monitoring(path):
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    monitor_path = 'selfdrive/ui/translations'
    logging.info('Started monitoring translation files.')
    start_monitoring(monitor_path)