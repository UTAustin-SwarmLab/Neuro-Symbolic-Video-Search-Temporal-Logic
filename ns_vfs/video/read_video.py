from pathlib import Path

import numpy as np

from ns_vfs.video.video import Video, VideoFormat


def read_video(
    video_path: str | Path | None = None,
    sequence_of_image: list[np.ndarray] | None = None,
) -> Video:
    """Read video from video_path or sequence of images.

    Args:
        video_path (str | Path | None): Path to video file. Defaults to None.
        sequence_of_image (list[np.ndarray] | None): Sequence of images
            as numpy arrays. Defaults to None.

    Returns:
        Video: Video object.

    Raises:
        ValueError: If neither or both video_path and
            sequence_of_image are provided.
    """
    if (video_path is None) == (sequence_of_image is None):
        msg = "Exactly one of video_path or sequence_of_image must be provided."
        raise ValueError(msg)
    if video_path:
        if isinstance(video_path, str):
            video_path = Path(video_path)

        if video_path.suffix == ".mp4":
            read_format = VideoFormat.MP4

    if sequence_of_image:
        read_format = VideoFormat.LIST_OF_ARRAY

    return Video(
        video_path=video_path,
        sequence_of_image=sequence_of_image,
        read_format=read_format,
    )
