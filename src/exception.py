import sys
# import traceback
# import logging
from src.logger import logging
# Function to get detailed error message
def error_msg_details(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occurred in python script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_msg

# Custom exception class
class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)  # Call the parent class constructor
        self.error_message = error_msg_details(error_msg, error_detail)

    def __str__(self):
        return self.error_message

# Example usage
if __name__=="__main__":
  try:
    a = 1 / 0
  except Exception as e:
    logging.info("Divide by zero error")
    raise CustomException(e, sys)
