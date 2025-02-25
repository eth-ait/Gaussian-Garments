# Add the parent directory of the project root to the Python path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from argparse import ArgumentParser
from post_processing import post_process
from dense_reconstruction import colmap
from src.datasets.prepare_4Ddress import Prepare4DDress

def build_parser():
    parser = ArgumentParser("Stage 1: Garment initialization.")
    parser.add_argument("--subject_path", "-s", required=True, type=str, help="The path to the subject folder that contains equence folders")
    parser.add_argument("--sequence_name", "-q", required=True, type=str, help="The name of the sequence dir, containing cameras.json")
    parser.add_argument("--garment_label", "-g", required=True, type=str)
    parser.add_argument("--output_path", "-o", required=False, type=str, help="The name of the output folder to be store results in.")
    parser.add_argument("--edge_len", default=0.01, type=float)
    parser.add_argument("--visualize", "-v", action='store_true')
    parser.add_argument("--camera", default="PINHOLE", type=str)
    parser.add_argument("--no_gpu", action='store_true')
    return parser

def main(args):
    dataset = Prepare4DDress(args.subject_path, args.sequence_name)
    colmap(dataset.output_root, args.camera, int(not args.no_gpu))
    post_process(dataset.output_root, args.garment_label, args.edge_len, args.visualize)

if __name__ == "__main__":
    # get arguments from command line
    parser = build_parser()
    args = parser.parse_args()
    main(args)
    

    
