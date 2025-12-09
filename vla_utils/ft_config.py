import json
from experiments.robot.libero.run_libero_eval import GenerateConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

class GenerateConfigFineTuned(GenerateConfig):
     
    pretrained_checkpoint_finetuned : str = "/home/romer-vla-sim/Workspace/fine_tuning_vlas/openvla-oft/runs_mustard40/1/openvla_7b+mustard_overfit+b2+lr-5e-05+lora-r16+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--5000_chkpt"
    model_variant : str = 'finetuned'

fp = "/home/romer-vla-sim/Workspace/inference_vla/openvla_oft/configs/training_cluster.json"
with open(fp, "r") as f:
    config = json.load(f)
    print("Loaded the config", fp.split("/")[-1].split(".")[0])


def get_config() -> GenerateConfigFineTuned:
    """
    Construct and return a GenerateConfig for OpenVLA-OFT.
    """
    cfg = GenerateConfigFineTuned(**config)
    cfg.pretrained_checkpoint_finetuned = config['pretrained_checkpoint'] 

    return cfg
