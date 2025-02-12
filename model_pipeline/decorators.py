import logging
import time

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def time_function(func):
    """Times the function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time  = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in "
              f"{execution_time:.4f} seconds.")
        return result
    return wrapper


def validate_model_loaded(func):
    """Ensures the model is loaded before making a prediction."""
    def wrapper(model, *args, **kwargs):
        if model is None:
            logging.error(
                "Model is not loaded. Please load the model first."
            )
            raise ValueError(
                "Model is not loaded. Please load the model first."
            )
        
        logging.info(f"Model is successfully loaded. Proceeding with " 
                     f"{func.__name__}...")
        return func(model, *args, **kwargs)
    return wrapper


def log_function_call(func):
    """Logs the arguments and return value of the function."""
    def wrapper(*args, **kwargs):
        # Log function arguments
        logging.info(
            f"Calling {func.__name__} with arguments: {args}, {kwargs}"
        )
        result = func(*args, **kwargs)
        # Log function return value
        logging.info(f"{func.__name__} returned: {result}")
        return result
    return wrapper