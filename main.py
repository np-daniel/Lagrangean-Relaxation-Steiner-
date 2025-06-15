from src import run_steiner

import logging
import argparse

def set_logger_configs():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler("tp1.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True)    
    return parser.parse_args()

def main():
    set_logger_configs()
    args = parse_args()
    for message in run_steiner(args.folder):
        logging.info(f"{message}")
    

if __name__ == "__main__":
    main()