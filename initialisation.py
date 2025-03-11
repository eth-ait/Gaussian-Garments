from pathlib import Path
from utils.defaults import DEFAULTS
from utils.preprocess_utils import PrepareDataset
from utils.initialisation_utils import COLMAP_recon, post_process
from argparse import ArgumentParser

def initialization_parser():
    parser = ArgumentParser("Stage 1: Garment initialization.")

    # Data Preprocessing & COLMAP reconstruction arguments
    parser.add_argument("--subject", "-s", required=True, type=str, help="Subject folder name that contains the sequence folders")
    parser.add_argument('--subject_out', '-so', type=str, default='')
    parser.add_argument("--sequence", "-q", required=True, type=str, help="The name of the sequence dir, containing cameras.json")
    parser.add_argument("--camera", default="PINHOLE", type=str)
    parser.add_argument("--no_gpu", action='store_true')

    # Post-processing arguments
    # parser.add_argument("--garment_type", "-g", required=True, type=str, help="The garment label to be processed, must be one of [upper, lower, outer].")
    parser.add_argument("--visualize", "-v", action='store_true')
    
    return parser
if __name__ == "__main__":
    # get arguments from command line
    parser = initialization_parser()
    args = parser.parse_args()

    if not args.subject_out:
        args.subject_out = args.subject
    
    source_root = Path(DEFAULTS.data_root) / args.subject / args.sequence
    target_root = Path(DEFAULTS.output_root) / args.subject_out / DEFAULTS.stage1
    dataset = PrepareDataset(source_root, target_root, args.camera)

    # COLMAP_recon(dataset.target_root, int(not args.no_gpu))
    post_process(dataset.target_root, args.visualize)