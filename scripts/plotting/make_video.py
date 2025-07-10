import os
import re

import imageio


def get_repeats(i, total):
    if i == 0:
        return 5
    if i / total < 0.5:
        return 2
    return 1


def make_video(mode, base_dir, output_dir, fps=2):
    """
    Creates a video by concatenating epoch images for the given mode.

    Parameters:
    - mode (str): One of 'train', 'eval', or 'update'.
    - base_dir (str): Root directory containing 'representations'.
    - output_dir (str or None): Directory to save the output video. Defaults to base_dir.
    - fps (int): Frames per second for the output video.
    """
    input_dir = os.path.join(base_dir, "representations", mode)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Directory {input_dir} does not exist")

    # Collect and sort epoch images
    files = [f for f in os.listdir(input_dir) if re.match(r"epoch_(\d+)\.png", f)]
    files_sorted = sorted(
        files, key=lambda fname: int(re.match(r"epoch_(\d+)\.png", fname).group(1))
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{mode}.mp4")

    # Write video
    writer = imageio.get_writer(output_path, fps=fps)
    for i, fname in enumerate(files_sorted):
        img = imageio.imread(os.path.join(input_dir, fname))
        repeats = get_repeats(i, len(files_sorted))
        for _ in range(repeats):
            writer.append_data(img)
    writer.close()

    print(f"Saved video to {output_path}")


# Generate videos for all modes
base_dir = "experiments/representations/2025-07-10-01-22-07"
output_dir = os.path.join("figures/videos", base_dir.split("/")[-1])
for mode in ["train", "eval", "update"]:
    make_video(mode, base_dir=base_dir, output_dir=output_dir, fps=4)
