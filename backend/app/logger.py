import logging
from logging.config import fileConfig
import os

def setup_logger(name):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'logging.conf')
    if os.path.exists(config_path):
        fileConfig(config_path)
    else:
        # 기본 로깅 설정
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)