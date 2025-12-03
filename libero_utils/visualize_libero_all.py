import h5py
import numpy as np
import os

data_dir = "/home/romer-vla-sim/Workspace/rlds_data/libero_rlds/libero_object_no_noops"

# Collect aggregate stats across all HDF5s
ee_pos_mean_abs_all, ee_pos_std_abs_all = [], []
ee_ori_mean_abs_all, ee_ori_std_abs_all = [], []
actions_mean_abs_all, actions_std_abs_all = [], []
demos_per_file = []

# Iterate over all .hdf5 files
for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".hdf5"):
        continue
    fpath = os.path.join(data_dir, fname)
    print(f"Processing {fname} ...")

    try:
        with h5py.File(fpath, "r") as h5:
            ee_pos_mean_abs, ee_pos_std_abs = [], []
            ee_ori_mean_abs, ee_ori_std_abs = [], []
            actions_mean_abs, actions_std_abs = [], []
            found_demos = []

            for i in range(50):
                demo_i = f"demo_{i}"
                if demo_i not in h5["data"]:
                    continue
                demo = h5["data"][demo_i]
                ee_pos = np.asarray(demo["obs"]["ee_pos"])
                ee_ori = np.asarray(demo["obs"]["ee_ori"])
                actions = np.asarray(demo["actions"])

                ee_pos_mean_abs.append(np.abs(ee_pos.mean(axis=0)))
                ee_pos_std_abs.append(np.abs(ee_pos.std(axis=0)))

                ee_ori_mean_abs.append(np.abs(ee_ori.mean(axis=0)))
                ee_ori_std_abs.append(np.abs(ee_ori.std(axis=0)))

                actions_mean_abs.append(np.abs(actions.mean(axis=0)))
                actions_std_abs.append(np.abs(actions.std(axis=0)))

                found_demos.append(demo_i)

            n_demos = len(found_demos)
            if n_demos == 0:
                print(f"  No demos found in {fname}")
                continue

            demos_per_file.append(n_demos)

            # Per-file mean over demos
            ee_pos_mean_abs_all.append(np.mean(ee_pos_mean_abs, axis=0) * n_demos)
            ee_pos_std_abs_all.append(np.mean(ee_pos_std_abs, axis=0) * n_demos)
            ee_ori_mean_abs_all.append(np.mean(ee_ori_mean_abs, axis=0) * n_demos)
            ee_ori_std_abs_all.append(np.mean(ee_ori_std_abs, axis=0) * n_demos)
            actions_mean_abs_all.append(np.mean(actions_mean_abs, axis=0) * n_demos)
            actions_std_abs_all.append(np.mean(actions_std_abs, axis=0) * n_demos)

            print(f"  {n_demos} demos processed.")

    except Exception as e:
        print(f"  Error reading {fname}: {e}")

# Combine stats across files weighted by number of demos
total_demos = sum(demos_per_file)
if total_demos > 0:
    print("\n=== Weighted averaged absolute mean/std across ALL tasks ===")
    print(f"Total demos processed: {total_demos}")

    ee_pos_mean_abs_final = sum(ee_pos_mean_abs_all) / total_demos
    ee_pos_std_abs_final  = sum(ee_pos_std_abs_all) / total_demos
    ee_ori_mean_abs_final = sum(ee_ori_mean_abs_all) / total_demos
    ee_ori_std_abs_final  = sum(ee_ori_std_abs_all) / total_demos
    actions_mean_abs_final = sum(actions_mean_abs_all) / total_demos
    actions_std_abs_final  = sum(actions_std_abs_all) / total_demos

    print("\nEE Position |mean|:", ee_pos_mean_abs_final)
    print("EE Position |std|:", ee_pos_std_abs_final)

    print("\nEE Orientation |mean|:", ee_ori_mean_abs_final)
    print("EE Orientation |std|:", ee_ori_std_abs_final)

    print("\nActions |mean|:", actions_mean_abs_final)
    print("Actions |std|:", actions_std_abs_final)
else:
    print("No valid HDF5 demos found.")
