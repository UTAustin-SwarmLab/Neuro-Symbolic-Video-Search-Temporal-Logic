from ns_vfs.generator.data_generator import WaymoLTLGroundTruthGenerator

DATA_DIR = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/waymo_benchmark_video"
SAVE_DIR = "/opt/Neuro-Symbolic-Video-Frame-Search/store/nsvs_artifact/waymo_banchmark_with_ltl_label"

if __name__ == "__main__":
    ltl_spec_generator = WaymoLTLGroundTruthGenerator(benchmark_ltl_frame_dir=DATA_DIR, save_dir=SAVE_DIR)
    ltl_spec_generator.generate()
