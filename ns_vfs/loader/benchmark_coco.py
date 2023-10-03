from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from ns_vfs.data.frame import BenchmarkRawImage
from ns_vfs.loader import LABEL_OF_INTEREST

from ._base import BenchmarkImageLoader


class COCOImageLoader(BenchmarkImageLoader):
    """Load COCO image data."""

    def __init__(self, coco_dir_path: str, annotation_file: str, image_dir: str):
        """Load COCO image data from file.

        Args:
        - coco_dir_path (str): Path to COCO dataset directory.
        - annotation_file (str): Name of the annotation file.
        - image_dir (str): Name of the image directory.
        """
        self.name = "COCO"
        self._coco_dir_path = Path(coco_dir_path)
        self._annotation_file = self._coco_dir_path / annotation_file
        self._image_dir = self._coco_dir_path / image_dir
        self._coco = COCO(self._annotation_file)
        self._original_class_labels = [cat["name"] for cat in self._coco.cats.values()]
        self.class_labels = [label for label in self._original_class_labels if label in LABEL_OF_INTEREST]
        self.images, self.annotations = self.load_data()
        self.data: BenchmarkRawImage = self.process_data()

    def load_data(self):
        """Load COCO image and annotation data.

        Returns:
        - images (list): A list of image file paths.
        - annotations (list): A list of annotation data corresponding to the images.
        """
        img_ids = self._coco.getImgIds()
        images = [self._image_dir / self._coco.loadImgs(id)[0]["file_name"] for id in img_ids]
        annotations = [self._coco.loadAnns(self._coco.getAnnIds(imgIds=id)) for id in img_ids]

        return images, annotations

    def process_data(self) -> BenchmarkRawImage:
        """Process raw COCO data to BenchmarkRawImage Data Class."""
        img_ids = self._coco.getImgIds()
        images = []
        class_labels = []
        # class_labels:
        # image_idx_1: ["label_1, label_2, ..."]
        # image_idx_2: ["label_1, label_2, ..."]
        for id in img_ids:
            images.append(
                cv2.imread(str(self._image_dir / self._coco.loadImgs(id)[0]["file_name"]))[:, :, ::-1]
            )  # Read it as RGB
            annotation = self._coco.loadAnns(self._coco.getAnnIds(imgIds=id))
            labels_per_image = []
            for i in range(len(annotation)):
                labels_per_image.append(self._coco.cats[annotation[i]["category_id"]]["name"])
            unique_labels = list(set(labels_per_image))
            if len(unique_labels) == 0:
                images.pop()
            else:
                class_labels.append(list(set(labels_per_image)))
        assert len(images) == len(class_labels)

        # Plot a sample image
        plt.imshow(images[5])
        plt.axis("off")
        plt.savefig("data_loader_sample_image.png")

        return BenchmarkRawImage(unique_labels=self.class_labels, labels=class_labels, images=images)

    def display_sample_image(self, index: int):
        """Display a sample image with annotations.

        Args:
        - index (int): The index of the sample image in the dataset.
        """
        image_path = str(self.images[index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = self._coco.getAnnIds(imgIds=self._coco.getImgIds()[index])
        anns = self._coco.loadAnns(ann_ids)
        plt.imshow(image)
        plt.axis("off")
        self._coco.showAnns(anns)
        plt.show()


# Example usage:
coco_loader = COCOImageLoader(
    coco_dir_path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/coco",
    annotation_file="annotations/instances_val2017.json",
    image_dir="val2017",
)

# Display a sample image
coco_loader.display_sample_image(0)
