import time


def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Execution time of '{func.__name__}': {elapsed_time:.6f} seconds")
        return result

    return wrapper
