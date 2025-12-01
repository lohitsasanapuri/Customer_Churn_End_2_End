import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)

    return logger