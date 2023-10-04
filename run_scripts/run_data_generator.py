from ns_vfs.generator.data_generator import BenchmarkVideoGenerator
from ns_vfs.loader.benchmark_cifar import Cifar10ImageLoader, Cifar100ImageLoader
from ns_vfs.loader.benchmark_coco import COCOImageLoader
from ns_vfs.loader.benchmark_imagenet import ImageNetDataloader

DATASET_TYPE = "coco"  # "cifar10" or "imagenet"

if __name__ == "__main__":
    if DATASET_TYPE == "cifar10":
        image_dir = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/cifar-10-batches-py"
        image_loader = Cifar10ImageLoader(cifar_dir_path=image_dir)
    elif DATASET_TYPE == "cifar100":
        image_dir = (
            "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/cifar-100-python"
        )
        image_loader = Cifar100ImageLoader(cifar_dir_path=image_dir)
    elif DATASET_TYPE == "imagenet":
        image_dir = "/store/datasets/ILSVRC"
        image_loader = ImageNetDataloader(imagenet_dir_path=image_dir)
    elif DATASET_TYPE == "coco":
        image_dir = "/store/datasets/ILSVRC"
        image_loader = COCOImageLoader(
            coco_dir_path="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/coco",
            annotation_file="annotations/instances_val2017.json",
            image_dir="val2017",
        )
    video_generator = BenchmarkVideoGenerator(
        image_data_loader=image_loader,
        artificat_dir="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/test_benchmark_frame_video",
    )
    ltl_logic_list = ["(prop1 & prop2) U prop3"]
    for ltl_logic in ltl_logic_list:
        video_generator.generate(max_number_frame=100, ltl_logic=ltl_logic, save_frames=False)
        # prop1 U prop2
        # F prop1
        # G prop1
        # (prop1 & prop2) U prop3
        print("Done!")
