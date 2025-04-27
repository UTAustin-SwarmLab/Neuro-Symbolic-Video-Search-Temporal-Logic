@dataclasses.dataclass
class FramesofInterest:
    """Frame class."""

    ltl_formula: str
    foi_list: List[List[int]] = dataclasses.field(default_factory=list)
    frame_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    annotated_images: List[np.ndarray] = dataclasses.field(default_factory=list)
    frame_idx_to_real_idx: dict = dataclasses.field(default_factory=dict)
    frame_buffer: List[Frame] = dataclasses.field(default_factory=list)

    def save_annotated_images(self, annotated_image: Dict[str, np.ndarray]):
        for a_img in list(annotated_image.values()):
            self.annotated_images.append(a_img)

    def save_frames_of_interest(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(self.frame_images):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(f"{path}/{idx}.png")
            try:
                if (
                    len(self.annotated_images) > 0
                    and self.annotated_images[idx] is not None
                ):
                    Image.fromarray(self.annotated_images[idx]).save(
                        f"{path}/{idx}_annotated.png"
                    )
            except:  # noqa: E722
                pass

    def flush_frame_buffer(self):
        """Flush frame buffer to frame of interest."""
        if len(self.frame_buffer) > 1:
            frame_interval = list()
            for frame in self.frame_buffer:
                frame_interval.append(frame.frame_idx)
                self.frame_idx_to_real_idx[frame.frame_idx] = frame.timestamp
                self.frame_images.append(frame.frame_image)
                self.save_annotated_images(frame.annotated_image)
            self.foi_list.append(frame_interval)
        else:
            for frame in self.frame_buffer:
                self.foi_list.append([frame.frame_idx])
                self.frame_idx_to_real_idx[frame.frame_idx] = frame.timestamp
                self.frame_images.append(frame.frame_image)
                self.save_annotated_images(frame.annotated_image)
        self.frame_buffer = list()

    def save(self, path: str | Path):
        from PIL import Image

        if isinstance(path, str):
            root_path = Path(path)
        else:
            root_path = path
        dir_name = get_file_or_dir_with_datetime("foi_result")
        frame_path = root_path / dir_name / "frame"
        annotation_path = root_path / dir_name / "annotation"

        frame_path.mkdir(parents=True, exist_ok=True)
        annotation_path.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(self.frame_images):
            Image.fromarray(img).save(f"{frame_path}/{idx}.png")
            try:
                if (
                    len(self.annotated_images) > 0
                    and self.annotated_images[idx] is not None
                ):
                    Image.fromarray(self.annotated_images[idx]).save(
                        f"{annotation_path}/{idx}_annotated.png"
                    )
            except:  # noqa: E722
                pass
