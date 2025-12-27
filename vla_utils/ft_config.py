import json
import os
import sys
from experiments.robot.libero.run_libero_eval import GenerateConfig

# --- HELPER FUNCTION TO FIX THE CHECKPOINT CONFIG ---
def fix_checkpoint_config_on_disk(checkpoint_path: str):
    """
    Checks the config.json in the checkpoint directory. 
    If 'auto_map' points to a remote repo (e.g., 'openvla/openvla...'), 
    it strips the prefix so transformers looks for local files.
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"[Warning] No config.json found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            
        
        changed = False
        if "auto_map" in data.keys():
            for key, value in data["auto_map"].items():
                # If the value contains the remote repo syntax "user/repo--filename"
                if "--" in value and "/" in value:
                    # Strip everything before the last "--" to get just the filename
                    # e.g., "openvla/openvla-7b--modeling_prismatic.OpenVLA..." -> "modeling_prismatic.OpenVLA..."
                    new_value = value.split("--")[-1]
                    data["auto_map"][key] = new_value
                    changed = True
                    print(f"[Auto-Fix] Changing {key} to local file: {new_value}")

        if changed:
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Success] Patched config.json at {config_path} to force local import.")
    except Exception as e:
        print(f"[Error] Could not patch config.json: {e}")

# --- YOUR CONFIG LOADER ---

#fp = "/home/romer-vla-sim/Workspace/inference_vla/openvla_oft/configs/mix_v2.json"
fp= "/dl_scratch1/romerhomerobotics/VLA-ZeroShot-Tests/configs/mix_v2_imagelab.json"


# Load the user config file
with open(fp, "r") as f:
    config_data = json.load(f)
    print("Loaded the config", fp.split("/")[-1].split(".")[0])

class GenerateConfigFineTuned(GenerateConfig):
    # We set defaults here, but they will be overwritten by the **config_data unpacking below
    pretrained_checkpoint_finetuned: str = config_data.get("pretrained_checkpoint", "")
    model_variant: str = 'finetuned'

def get_config() -> GenerateConfigFineTuned:
    """
    Construct and return a GenerateConfig for OpenVLA-OFT.
    """
    # 1. FIX THE CHECKPOINT ON DISK BEFORE LOADING
    # This ensures modeling_prismatic.py is loaded from the local folder, not ~/.cache
    # if "pretrained_checkpoint" in config_data:
        # fix_checkpoint_config_on_disk(config_data["pretrained_checkpoint"])
    
    # 2. Instantiate the config object
    # We filter config_data to only pass keys that GenerateConfigFineTuned expects if necessary,
    # but typically kwargs unpacking works if the parent class accepts extras or you strictly match keys.
    cfg = GenerateConfigFineTuned(**config_data)
    
    # 3. Explicitly map the checkpoint path
    #cfg.pretrained_checkpoint_finetuned = config_data['pretrained_checkpoint'] 

    return cfg