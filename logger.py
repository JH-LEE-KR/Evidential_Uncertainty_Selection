"""
# * Original Code
# * https://github.com/microsoft/Swin-Transformer/blob/main/logger.py
# * Modification:
# * Remove unnecessary parts for my model and add code for my logging
"""

import os
import sys
import logging
import functools
from termcolor import colored
import datetime

@functools.lru_cache()
def create_logger(output_path, name=''):
    file_name = datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M')

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_path, '{}.txt'.format(file_name)), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger, file_name