from utils.arg_utils import initialization_parser
from utils.preprocess_utils import PrepareDataset
from utils.initialisation_utils import COLMAP_recon, post_process

if __name__ == "__main__":
    # get arguments from command line
    parser = initialization_parser()
    args = parser.parse_args()
    
    dataset = PrepareDataset(args.subject, args.sequence, args.camera)

    COLMAP_recon(dataset.output_root, args.camera, int(not args.no_gpu))

    post_process(dataset.output_root, args.garment_type, args.visualize)