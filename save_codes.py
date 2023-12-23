"""
import torch
import logging

# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Log information about the available device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.GPU_ID}")
        logging.info(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        logging.warning("CUDA is not available. Using CPU.")
"""