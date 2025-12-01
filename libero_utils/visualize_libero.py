import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio # Add this line for video writing

# --- File Paths and Data Loading (Existing Code) ---
hdf5_file = "/home/romer-vla-sim/Workspace/rlds_data/libero_rlds/libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"

h5 = h5py.File(hdf5_file, "r")
demo1 = h5["data"]["demo_9"]
steps = 0
count = 0
for i in range(50):
    try:
        demo_i = "demo_" + str(i)
        demo = h5["data"][demo_i]
        count +=1
        ee_states = np.asarray(demo["obs"]["ee_states"])
        steps += ee_states.shape[0]
    except Exception:
        pass
print(steps / count)


# Load datasets
images = np.asarray(demo1["obs"]["agentview_rgb"])[:,::-1,::-1]
actions = np.asarray(demo1["actions"])
ee_pos = np.asarray(demo1["obs"]["ee_pos"])
ee_ori = np.asarray(demo1["obs"]["ee_ori"])
ee_states = np.asarray(demo1["obs"]["ee_states"])
steps = np.asarray(demo["obs"]["ee_states"]).shape[0]

print("EE Position mean:", ee_pos.mean(axis=0), "std:", ee_pos.std(axis=0))
print("EE Orientation mean:", ee_ori.mean(axis=0), "std:", ee_ori.std(axis=0))
print("Actions mean:", actions.mean(axis=0), "std:", actions.std(axis=0))

# --- Video Setup ---
# Define the output file name and frames per second (FPS)
OUTPUT_VIDEO_FILE = "demo_visualization.mp4"
FPS = 20 # You can adjust this value

# Visualization
plt.ioff() # Use ioff for video generation
fig, ax = plt.subplots(figsize=(4, 4)) # Adjust figure size for 256x256 image aspect
num_frames = images.shape[0]

# --- Collecting Frames for Video ---
# Create a list to store the rendered frames as numpy arrays
video_frames = []

for i in range(num_frames):
    ax.clear()
    img = images[i]

    # Overlay EE position
    x, y, z = ee_pos[i]
    ax.imshow(img)

    # Overlay EE pos and ori as text
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
    
    # --- Save the current figure to a numpy array ---
    fig.canvas.draw()
    # Convert the figure to an array and append to the list
    frame_array = np.array(fig.canvas.renderer.buffer_rgba())
    # Keep only the RGB channels (drop the alpha channel)
    video_frames.append(frame_array[..., :3]) 
    
# Close the plot figure after the loop finishes
plt.close(fig)

# --- Save the Video File ---
print(f"Saving video to {OUTPUT_VIDEO_FILE}...")
iio.imwrite(
    OUTPUT_VIDEO_FILE, 
    video_frames, 
    fps=FPS, 
    quality=8, # 0 (lowest) to 10 (highest) quality. 
    codec='libx264' # Common codec for MP4
)
print("Video saved successfully.")
