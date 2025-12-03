import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import random

# --- Parameters ---
data_dir = "/home/romer-vla-sim/Workspace/rlds_data/libero_rlds/libero_object_no_noops"
OUTPUT_DIR = "./demo_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 20  # frames per second
NUM_DEMOS_PER_TASK = 3
NUM_TASKS = 3  # first 3 HDF5 files (can adjust)

# --- Get HDF5 files ---
hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
hdf5_files = hdf5_files[:NUM_TASKS]  # limit to 3 tasks

# --- Function to generate video for a single demo ---
def generate_video(images, ee_pos, ee_ori, actions, output_file, fps=20):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(4, 4))
    video_frames = []

    for i in range(images.shape[0]):
        ax.clear()
        img = images[i]
        ax.imshow(img)

        # Overlay EE pos, ori, and actions
        x, y, z = ee_pos[i]
        ori = ee_ori[i]
        act = actions[i]
        text = (
            f"Pos: [{x:.2f}, {y:.2f}, {z:.2f}]\n"
            f"Ori: [{ori[0]:.2f}, {ori[1]:.2f}, {ori[2]:.2f}]\n"
            f"Action: [{', '.join(f'{a:.2f}' for a in act)}]"
        )
        ax.text(5, 20, text, color="yellow", fontsize=10, backgroundcolor="black")
        ax.set_title(f"Frame {i}")
        ax.axis("off")

        fig.canvas.draw()
        frame_array = np.array(fig.canvas.renderer.buffer_rgba())
        video_frames.append(frame_array[..., :3])

    plt.close(fig)
    iio.imwrite(output_file, video_frames, fps=fps, quality=8, codec='libx264')
    print(f"Saved video: {output_file}")

# --- Main loop: iterate over tasks and demos ---
for task_idx, fname in enumerate(hdf5_files):
    fpath = os.path.join(data_dir, fname)
    print(f"\nProcessing task {task_idx+1}: {fname}")

    with h5py.File(fpath, "r") as h5:
        available_demos = [k for k in h5["data"].keys()]
        if len(available_demos) < NUM_DEMOS_PER_TASK:
            print(f"  Warning: only {len(available_demos)} demos available. Adjusting selection.")
            selected_demos = available_demos
        else:
            selected_demos = random.sample(available_demos, NUM_DEMOS_PER_TASK)

        for demo_idx, demo_key in enumerate(selected_demos):
            demo = h5["data"][demo_key]
            images = np.asarray(demo["obs"]["agentview_rgb"])[:, ::-1, ::-1]
            actions = np.asarray(demo["actions"])
            ee_pos = np.asarray(demo["obs"]["ee_pos"])
            ee_ori = np.asarray(demo["obs"]["ee_ori"])

            output_file = os.path.join(
                OUTPUT_DIR, f"task{task_idx+1}_{demo_key}.mp4"
            )
            generate_video(images, ee_pos, ee_ori, actions, output_file, fps=FPS)

print("\nAll videos generated successfully.")
