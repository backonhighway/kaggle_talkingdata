import logging


def get_my_logger():
    log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    logging.basicConfig(format=log_fmt,
                        datefmt='%Y-%m-%d/%H:%M:%S',
                        filename='../logs/temp.log',
                        level='INFO')
    logger = logging.getLogger(__name__)
    return logger
