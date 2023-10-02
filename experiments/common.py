from pathlib import Path


def get_available_benchmark_video(path_to_directory: str):
    if isinstance(path_to_directory, str):
        directory_path = Path(path_to_directory)
        return list(directory_path.glob("*.pkl"))
    else:
        directory_path = path_to_directory
        return list(directory_path.rglob("*.pkl"))
