import logging
from datetime import datetime


class CustomLogger:
    def __init__(self, filename='testlogs.txt'):
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setLevel(logging.DEBUG)

        # Create formatter
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.formatter)

        # Add file handler to logger
        self.logger.addHandler(self.file_handler)

    def log(self, message, level='INFO'):
        # Log the message with specified level
        if level == 'DEBUG':
            self.logger.debug(message)
        elif level == 'INFO':
            self.logger.info(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        elif level == 'CRITICAL':
            self.logger.critical(message)


if __name__ == '__main__':
    logger = CustomLogger()
    # Log messages
    logger.log('This is an info message')
    logger.log('This is a warning message', level='WARNING')
    logger.log('This is an error message', level='ERROR')
