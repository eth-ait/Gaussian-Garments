import subprocess
import time
from utils.defaults import DEFAULTS
import argparse
from pathlib import Path


def check_if_template(tdir):
    tdir = Path(tdir)
    obj_files = list(tdir.glob('*_uv.obj'))
    ply_files = list(tdir.glob('*.ply'))

    return len(obj_files) > 0 and len(ply_files) > 0


    print()
    print('tdir', tdir)
    print('obj_files', obj_files)
    print('ply_files', ply_files)



def tt_00136_outer_f(take):
    if take in ['Take17', 'Take18']:
        return 'Take17'
    else:
        return 'Take18'
    

def tt_00176_outer_f(take):
    if take in ['Take9', 'Take10']:
        return 'Take10'
    else:
        return 'Take9'

def tt_10004_outer_f(take):
    if take in ['Take7', 'Take0']:
        return 'Take0'
    else:
        return 'Take7'
    
# def tt_00152_outer_f(take):
#     if take in ['Take7', 'Take0']:
#         return 'Take0'
#     else:
#         return 'Take7'



modified_closs_paths = dict()
modified_closs_paths['00136_outer'] = tt_00136_outer_f
modified_closs_paths['00176_outer'] = tt_00176_outer_f



modified_closs_paths['10004_outer'] = tt_10004_outer_f



# modified_closs_paths['00152_outer'] = dict()

def get_template_take(sname, gname, garment, take):
    gstr = f"{sname}_{garment}"
    if gstr in modified_closs_paths:
        return modified_closs_paths[gstr](take)
    else:
        templates_path = Path('../datas/')
        template_dirs_seq = templates_path.glob(f'{sname}_{gname}_*')
        template_dirs_seq = list(template_dirs_seq)

        for tdir in template_dirs_seq:
            if 'Old' in tdir.name:
                continue
            if check_if_template(tdir):
                template_dir = tdir
                break
        _, _, template_take = template_dir.name.split('_')

        return template_take



def main():
    parser = argparse.ArgumentParser(description='Register Meta')
    parser.add_argument('-s', type=str, help='source path.')
    parser.add_argument('-g', type=str, help='garment type.')
    parser.add_argument('-t', type=str, help='registration type: VB / body / VB_body / NoVol / NoPhys')
    
    args = parser.parse_args()

    registration_root = Path(DEFAULTS.registration_root)
    data_root = Path(DEFAULTS.dataset_root)

    
    sname, gname, take = args.s.split('/')


    # templates_path = Path('../datas/')
    # template_dirs_seq = templates_path.glob(f'{sname}_{gname}_*')
    # template_dirs_seq = list(template_dirs_seq)

    # for tdir in template_dirs_seq:
    #     if check_if_template(tdir):
    #         template_dir = tdir
    #         break

    # template_dir = template_dirs_seq[0]
    # print('template_dirs_seq', template_dirs_seq)

    # assert False

    # _, _, template_take = template_dir.name.split('_')

    template_take = get_template_take(sname, gname, args.g, take)
    is_template_take = take == template_take

    gseq = f"{sname}_{args.g}"

    out_path = Path(args.t) / f"{sname}_{args.g}" / take.lower()


    if is_template_take:
        cross_from = ""
        script_file = 'train.py'
    else:
        script_file = "cross_register.py"

        if sname == '00176' and args.g == 'lower':
            cross_path = Path(args.t) / f"00176_lower" / 'take17'
        else:
            cross_path = Path(args.t) / f"{sname}_{args.g}" / template_take.lower()
            
        cross_from = f"-c {cross_path}"
        is_template_ready_path = registration_root / cross_path / 'meshes'


    if args.t == 'VB':
        collision_iteration = 0
        ci_arg = f'--collision_iteration {collision_iteration}'
        vol_arg = ''
    elif args.t == 'body':
        collision_iteration = 5000
        ci_arg = f'--collision_iteration {collision_iteration}'
        vol_arg = ''
    elif args.t == 'VB_body':
        collision_iteration = 2000
        ci_arg = f'--collision_iteration {collision_iteration}'
        vol_arg = ''
    elif args.t == 'NoVol':
        ci_arg = ''
        vol_arg = '--no_volume'
    elif args.t == 'NoPhys':
        ci_arg = ''
        vol_arg = '--no_physics'
    else:
        raise ValueError(f"Unknown registration type: {args.t}")

    source_dir = Path(DEFAULTS.dataset_root) / sname / gname  / take
    registration_dir = Path(DEFAULTS.registration_root) / args.t / gseq / take.lower()
    meshes_dir = registration_dir / 'meshes'
    smplx_dir = source_dir / 'smplx'

    if meshes_dir.exists():
        n_target = len(list(meshes_dir.glob('*.obj')))
        n_source = len(list(smplx_dir.glob('*.pkl')))

        print(f"n_target: {n_target}, n_source: {n_source}")
        if n_target == n_source:
            print('already registered')
            return


    if not is_template_take:
        while True:
            if is_template_ready_path.exists() and len(list(is_template_ready_path.glob('*.obj'))) > 0:
                break
            print(f'waiting for template  {is_template_ready_path} to be ready')
            time.sleep(10)
    command = f"python {script_file} -s {args.s} -g {args.g} -m  {out_path} {ci_arg} {cross_from} {vol_arg} --eval"

    if sname == '00190' and gname == 'Inner' and take == 'Take9':
        command += ' --other_frame_iterations 7000'

    print('RUNNING:', command)
    command = command.split()
    result = subprocess.run(command)



if __name__ == "__main__":
    main()