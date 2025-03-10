import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    '''returns an error string with the filename and line number where the error occurred'''
    # sys.exc_info() -> returns a tuple with info about curr exception being handled
    # exc_tb = traceback information -> contains info about filename and line number
    exc_type, exc_val, exc_tb = sys.exc_info()
    # retrieves filename where exception occurred
    file_name= exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f'{error} - File "{file_name}, line {line_number}" - {exc_type.__name__}: {exc_val}'
    logging.info(error_message)
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # constructor of superclass called to initialize the custom exception
        # custom exception inherits the basic properties and behavior of regular exceptions, while still allowing us you to customize it as needed. 
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message