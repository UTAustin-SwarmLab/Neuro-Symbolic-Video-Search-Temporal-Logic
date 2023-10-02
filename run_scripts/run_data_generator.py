from ns_vfs.generator.data_generator import BenchmarkVideoGenerator
<<<<<<< HEAD
from ns_vfs.loader.benchmark_image import Cifar10ImageLoader, ImageNetDataloader


DATASET_TYPE = "imagenet" # "cifar10" or "imagenet"
if __name__ == "__main__":
    if DATASET_TYPE == "cifar10":
        image_dir = (
            "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/cifar-10-batches-py"
        )
        image_loader = Cifar10ImageLoader(cifar_dir_path=image_dir)
    elif DATASET_TYPE == "imagenet":
        image_dir = "/store/datasets/ILSVRC"
        image_loader = ImageNetDataloader(imagenet_dir_path=image_dir)
    
=======
from ns_vfs.loader.benchmark_image import Cifar10ImageLoader

if __name__ == "__main__":
    cifar_dir = (
        "/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/data/benchmark_image_dataset/cifar-10-batches-py"
    )
    image_loader = Cifar10ImageLoader(cifar_dir_path=cifar_dir)
>>>>>>> a6438faaa43390dec3a3aa166812f22f34b54869
    cifar_video_generator = BenchmarkVideoGenerator(
        image_data_loader=image_loader,
        artificat_dir="/opt/Neuro-Symbolic-Video-Frame-Search/artifacts/benchmark_frame_video",
    )
    cifar_video_generator.generate(max_number_frame=100, ltl_logic="prop1 U prop2", save_frames=False)
    # prop1 U prop2
    # F prop1
    # G prop1
    print("Done!")
