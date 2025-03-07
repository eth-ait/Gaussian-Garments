from pathlib import Path
from utils.arg_utils import initialization_parser
from utils.defaults import DEFAULTS
from utils.preprocess_utils import PrepareDataset
from utils.initialisation_utils import COLMAP_recon, post_process

if __name__ == "__main__":
    # get arguments from command line
    parser = initialization_parser()
    args = parser.parse_args()
    
    source_root = Path(DEFAULTS.data_root) / args.subject / args.sequence
    target_root = Path(DEFAULTS.output_root) / args.subject / DEFAULTS.stage1
    dataset = PrepareDataset(source_root, target_root, args.camera)

    COLMAP_recon(dataset.target_root, int(not args.no_gpu))
    post_process(dataset.target_root, args.garment_type, args.visualize)