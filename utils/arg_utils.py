import os
from argparse import ArgumentParser

def initialization_parser():
    parser = ArgumentParser("Stage 1: Garment initialization.")

    # Data Preprocessing & COLMAP reconstruction arguments
    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders")
    parser.add_argument("--sequence", "-q", required=True, type=str, help="The name of the sequence dir, containing cameras.json")
    parser.add_argument("--camera", default="PINHOLE", type=str)
    parser.add_argument("--no_gpu", action='store_true')

    # Post-processing arguments
    parser.add_argument("--garment_type", "-g", required=True, type=str, help="The garment label to be processed, must be one of [upper, lower, outer].")
    parser.add_argument("--visualize", "-v", action='store_true')
    
    return parser