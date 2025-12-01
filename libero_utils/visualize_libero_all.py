import h5py
import numpy as np

hdf5_file = "/home/romer-vla-sim/Workspace/rlds_data/libero_rlds/libero_object_no_noops/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"

with h5py.File(hdf5_file, "r") as h5:
    ee_pos_list = []
    ee_ori_list = []
    actions_list = []
    steps_list = []   # per-demo step counts
    found_demos = []

    for i in range(50):
        demo_i = f"demo_{i}"
        try:
            demo = h5["data"][demo_i]
            ee_pos = np.asarray(demo["obs"]["ee_pos"])
            ee_ori = np.asarray(demo["obs"]["ee_ori"])
            actions = np.asarray(demo["actions"])

            n_steps = ee_pos.shape[0]
            steps_list.append(n_steps)

            ee_pos_list.append(ee_pos)
            ee_ori_list.append(ee_ori)
            actions_list.append(actions)
            found_demos.append(demo_i)

            # print(f"Loaded {demo_i}: {n_steps} steps")

        except KeyError:
            # demo not present, skip
            # print(f"Skipping {demo_i} (not found)")
            continue

    if len(found_demos) == 0:
        print("No demos found in the file.")
    else:
        # Concatenate all available demos (concatenate along timestep axis)
        ee_pos_all = np.concatenate(ee_pos_list, axis=0)
        ee_ori_all = np.concatenate(ee_ori_list, axis=0)
        actions_all = np.concatenate(actions_list, axis=0)

        # Aggregated stats over all timesteps (correct population std)
        print("\n=== Aggregated statistics over ALL timesteps (all demos combined) ===")
        print("Total demos found:", len(found_demos))
        print("Total timesteps:", ee_pos_all.shape[0])

        print("\nEE Position mean:", ee_pos_all.mean(axis=0))
        print("EE Position std:", ee_pos_all.std(axis=0))

        print("\nEE Orientation mean:", ee_ori_all.mean(axis=0))
        print("EE Orientation std:", ee_ori_all.std(axis=0))

        print("\nActions mean:", actions_all.mean(axis=0))
        print("Actions std:", actions_all.std(axis=0))

        # # Per-demo step statistics
        # steps_arr = np.asarray(steps_list, dtype=int)
        # print("\n=== Per-demo step statistics ===")
        # for demo_name, n in zip(found_demos, steps_list):
        #     print(f"{demo_name}: {n} steps")

        # print("\nSummary of demo lengths (timesteps):")
        # print("Total steps across demos:", steps_arr.sum())
        # print("Number of demos (present):", steps_arr.size)
        # print("Mean steps per demo:", steps_arr.mean())
        # print("Std of steps per demo:", steps_arr.std())   # population std
        # print("Min steps in a demo:", steps_arr.min())
        # print("Max steps in a demo:", steps_arr.max())

