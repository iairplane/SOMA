# Script for sampling initial states
# This script creates an OffScreenRenderEnv environment, forcing random generation 
# of initial states upon reset(), and samples a specified number of states.
# Target path: ~/libero/scripts/sample_init_states.py

import argparse
import os
import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T

def sample_init_states(bddl_file, num_samples, save_path):
    print(f"Sampling {num_samples} initial states for {bddl_file}...")
    
    # Create the environment, bypassing preset init states so that 
    # new states are randomly generated upon calling reset()
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        control_freq=20,
    )

    init_states = []
    
    for i in range(num_samples):
        try:
            env.reset()
            # Retrieve the current internal state of the simulator (xml, data, etc.)
            # get_state() is a low-level API of robosuite/libero
            state = env.sim.get_state().flatten()
            init_states.append(state)
            print(f"[Success] Sampled state {i+1}/{num_samples}")
        except Exception as e:
            print(f"[Error] Failed to sample state {i+1}: {e}")

    # Convert the collected states to a numpy array
    init_states = np.array(init_states)
    
    # Save in Libero's standard format
    # Official init files are typically torch.Tensor or numpy arrays.
    # Here, we save it as a torch tensor to maintain compatibility.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(torch.from_numpy(init_states), save_path)
    
    print(f"Done! Saved to: {save_path}")
    print(f"State shape: {init_states.shape}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample random initial states for a given BDDL task.")
    parser.add_argument("--bddl_file", type=str, required=True, help="Absolute or relative path to the BDDL file.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of initial states to sample.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated .init file.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bddl_file):
        raise FileNotFoundError(f"BDDL file not found: {args.bddl_file}")
        
    sample_init_states(args.bddl_file, args.num_samples, args.save_path)
