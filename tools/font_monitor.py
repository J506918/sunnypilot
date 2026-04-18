import logging
import time
import os

# Setup logging
logging.basicConfig(filename='font_monitor.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def monitor_font(font_path):
    retries = 3
    for attempt in range(retries):
        try:
            # Simulated monitor function
            logging.info(f'Monitoring font at {font_path}')
            # Placeholder for actual monitoring logic
            if not os.path.exists(font_path):
                raise FileNotFoundError(f'Font file not found: {font_path}')
            # More logic goes here...
            
            logging.info(f'Successfully monitored font: {font_path}')
            return  # Exit upon success
            
        except Exception as e:
            logging.error(f'Error monitoring font: {e}')
            if attempt < retries - 1:
                logging.info(f'Retrying... ({attempt + 1}/{retries})')
                time.sleep(2)  # Wait before retrying
            else:
                logging.error('Max retries reached. Exiting.')

if __name__ == "__main__":
    font_files = ["font1.ttf", "font2.ttf", "font3.ttf"]  # Example font files
    for font_file in font_files:
        monitor_font(font_file)