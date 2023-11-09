from ns_vfs.generator.data_generator import BenchmarkVideoGenerator
from ns_vfs.loader.benchmark_cifar import Cifar10ImageLoader, Cifar100ImageLoader
from ns_vfs.loader.benchmark_coco import COCOImageLoader
from ns_vfs.loader.benchmark_imagenet import ImageNetDataloader

DATASET_TYPE = "coco"  # "cifar10" or "imagenet" or "coco"
MODE = "until_time_delta"  # until_time_delta
if __name__ == "__main__":
    if DATASET_TYPE == "cifar10":
        image_dir = "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/cifar-10-batches-py"
        image_loader = Cifar10ImageLoader(cifar_dir_path=image_dir)
    elif DATASET_TYPE == "cifar100":
        image_dir = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/data/benchmark_image_dataset/cifar-100-python"
        image_loader = Cifar100ImageLoader(cifar_dir_path=image_dir)
    elif DATASET_TYPE == "imagenet":
        image_dir = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/data/ILSVRC"
        image_loader = ImageNetDataloader(imagenet_dir_path=image_dir)
    elif DATASET_TYPE == "coco":
        image_loader = COCOImageLoader(
            coco_dir_path="/opt/Neuro-Symbolic-Video-Frame-Search/store/datasets/COCO/2017",
            annotation_file="annotations/instances_val2017.json",
            image_dir="val2017",
        )
    video_generator = BenchmarkVideoGenerator(
        image_data_loader=image_loader,
        artificat_dir="/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/timedelta_benchmark_frame_video_generated",
    )
    if MODE == "regular":
        """
        SPECIFICATION BELOW ONLY WORKS WITH COCO DATASET
        prop1 & prop2
        (prop1 & prop2) U prop3
        """
        ltl_logic_list = ["F prop1", "G prop1", "prop1 U prop2", "(prop1 & prop2) U prop3", "prop1 & prop2"]
        for ltl_logic in ltl_logic_list:
            video_generator.generate(
                initial_number_of_frame=25,
                max_number_frame=50,  # 500
                ltl_logic=ltl_logic,
                save_frames=False,
                number_video_per_set_of_frame=2,
            )
            # prop1 U prop2
            # prop1 & prop2
            # F prop1
            # G prop1
            # (prop1 & prop2) U prop3
            print("Done!")
    elif MODE == "until_time_delta":
        video_generator.generate_until_time_delta(
            max_number_frame=2400,
            number_video_per_set_of_frame=1,
            present_prop1_till_prop2=True,
            increase_rate=10,
            initial_number_of_frame=10,
        )
