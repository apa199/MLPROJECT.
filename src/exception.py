import sys
from src.logger import logging  # Make sure logger.py exists

def error_details(error, error_detail: sys):
    """
    Returns a formatted error message with file name and line number.
    """
    # Get the traceback object
    _, _, exc_tb = error_detail.exc_info()  # <-- traceback object
    
    # File where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a formatted error message
    error_message = "Issue in Python script [{0}], line number [{1}], error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        # Generate the formatted error message
        self.error_message = error_details(error, error_detail)
        
        # Call base Exception constructor
        super().__init__(self.error_message)
        
        # Optional: log the error
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Attempted invalid operation")
        raise CustomException(e, sys)
