def error_message_detail(error, error_detail):
    '''returns an error string with the filename and line number where the error occurred'''
    # sys.exc_info() -> returns a tuple with info about curr exception being handled
    # exc_tb = traceback information -> contains info about filename and line number
    exc_type, _, exc_tb = error_detail.exc_info()
    # retrieves filename where exception occurred
    file_name= exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"{exc_type} occured in [{file_name}] at line number [{line_number}]: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        # constructor of superclass called to initialize the custom exception
        # custom exception inherits the basic properties and behavior of regular exceptions, while still allowing us you to customize it as needed. 
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message