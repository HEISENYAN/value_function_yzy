from huggingface_hub import snapshot_download
import zipfile
import os
import shutil
from pathlib import Path
from robotwin_to_lerobot import port_aloha_direct

TASK = [
    "adjust_bottle",
    "place_can_basket", 
    "beat_block_hammer",
    "place_cans_plasticbox",
    "blocks_ranking_rgb",
    "place_container_plate",
    "blocks_ranking_size",
    "place_dual_shoes",
    "click_alarmclock",
    "place_empty_cup",
    "click_bell",
    "place_fan",
    "dump_bin_bigbin",
    "place_mouse_pad",
    "grab_roller",
    "place_object_basket",
    "handover_block",
    "place_object_scale",
    "handover_mic",
    "place_object_stand",
    "hanging_mug",
    "place_phone_stand",
    "lift_pot",
    "place_shoe",
    "move_can_pot",
    "press_stapler",
    "move_pillbottle_pad",
    "put_bottles_dustbin",
    "move_playingcard_away",
    "put_object_cabinet",
    "move_stapler_pad",
    "rotate_qrcode",
    "open_laptop",
    "scan_object",
    "open_microwave",
    "shake_bottle",
    "pick_diverse_bottles",
    "shake_bottle_horizontally",
    "pick_dual_bottles",
    "stack_blocks_three",
    "place_a2b_left",
    "stack_blocks_two",
    "place_a2b_right",
    "stack_bowls_three",
    "place_bread_basket",
    "stack_bowls_two",
    "place_bread_skillet",
    "stamp_seal",
    "place_burger_fries",
    "turn_switch"
]

for task in TASK:
    snapshot_download(
        repo_id="TianxingChen/RoboTwin2.0",
        allow_patterns=[f"dataset/{task}/aloha-agilex_randomized_50.zip"],
        local_dir="../data",
        repo_type="dataset",
        resume_download=True,
    )
    
    zip_path = f"../data/dataset/{task}/aloha-agilex_randomized_50.zip"
    extract_path = f"../data/dataset/{task}"
    
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {task} zip file")
        
        os.remove(zip_path)
        print(f"Removed {task} zip file")
        
        raw_dir = Path(f"../data/dataset/{task}/aloha-agilex_randomized_50")
        repo_id = f"robotwin_aloha_lerobot/{task}_randomized_50"
        
        try:
            port_aloha_direct(
                raw_dir=raw_dir,
                repo_id=repo_id,
                mode="video",
                push_to_hub=True
            )
            print(f"Successfully processed {task}")
            
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)
                print(f"Removed raw data directory for {task}")
                
        except Exception as e:
            print(f"Error processing {task}: {e}")
    else:
        print(f"Zip file not found for {task}")