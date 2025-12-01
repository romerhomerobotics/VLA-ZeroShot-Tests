from experiments.robot.libero.run_libero_eval import GenerateConfig
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

class GenerateConfigFineTuned(GenerateConfig):
     
    pretrained_checkpoint_finetuned : str =  "/home/romer-vla-sim/Workspace/fine_tuning_vlas/openvla-oft/runs/openvla_7b+train_with_proprio+b2+lr-0.0005+lora-r16+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt" 
    model_variant : str = 'finetuned'



def get_config() -> GenerateConfigFineTuned:
    """
    Construct and return a GenerateConfig for OpenVLA-OFT.
    """
    cfg = GenerateConfigFineTuned(
        pretrained_checkpoint="/home/romer-vla-sim/Workspace/fine_tuning_vlas/openvla-oft/runs/openvla_7b+train_with_proprio+b2+lr-0.0005+lora-r16+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt",
        #"/home/romer-vla-sim/Workspace/fine_tuning_vlas/openvla-oft/run_no_proprio/openvla_7b+train+b2+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--150000_chkpt", 
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=False,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=8,
        unnorm_key="mustard_pick_place_train",
    )

    return cfg

