from datetime import datetime


def get_filename_with_datetime(base_name):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{current_time}.png"
