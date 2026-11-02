"""Create a portable run and launch preparation, baseline or SMA training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def initialize(args):
    root = Path(args.run).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new empty run directory.')
    openpi, checkpoint, data = map(lambda s: Path(s).resolve(), [args.openpi, args.checkpoint, args.data])
    cfg = json.loads((HERE / 'configs/common.json').read_text())
    head = subprocess.check_output(['git', '-C', str(openpi), 'rev-parse', 'HEAD'], text=True).strip()
    if head != cfg['source_commit']:
        raise ValueError('OpenPI commit differs from the reference.')
    if 'return_r8_components' not in (openpi / 'src/openpi/models/pi0.py').read_text():
        raise ValueError('Run setup_openpi.py to install the optional residual hook.')
    if not (checkpoint / 'params').is_dir():
        raise ValueError('Expected a complete official Pi0.5 checkpoint directory with params/.')
    for name in ['EPISODES_1500.json', 'TRAIN_STATS.json', f'TRAIN_WINDOWS_seed{args.seed}_G64_U10000.npy', 'numeric_windows.npz']:
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    cfg.update(c0=str(checkpoint), output=str(root / 'output'), r8_root=str(root),
               seed=args.seed, data_seed=args.seed, method='S2' if args.method == 'sma' else 'U_B')
    for folder in ['configs', 'code', 'receipts', 'cache/structure']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / 'code/openpi').symlink_to(openpi, target_is_directory=True)
    (root / 'data_manifests').symlink_to(data, target_is_directory=True)
    for name in ['CHAIN', 'SOURCE_TIME', 'PI05']:
        (root / 'configs' / (name + '.json')).write_text(json.dumps(cfg, indent=2) + '\n')
    (root / 'configs/PATHS.json').write_text(json.dumps({'remote_data': str(root)}) + '\n')
    shutil.copy2(data / 'numeric_windows.npz', root / 'cache/structure/numeric_windows.npz')
    print(root)


def launch(args):
    root = Path(args.run).resolve()
    cfg = json.loads((root / 'configs/CHAIN.json').read_text())
    if args.command == 'prepare':
        if cfg['method'] != 'S2':
            raise ValueError('Uniform baseline does not require frozen features or graph preparation.')
        script = 'prepare_structure.py'
    else:
        if cfg['method'] == 'S2' and not (root / 'cache/structure/PI_STRUCTURE_COMPLETE.json').exists():
            raise FileNotFoundError('Run the SMA frozen-feature/graph preparation first.')
        script = 'train_pi05_adaptive.py' if cfg['method'] == 'S2' else 'train_pi05.py'
    env = dict(os.environ, SMA_RUN_DIR=str(root))
    env.setdefault('R5_STOP_FILE', str(root / 'STOP'))
    env.setdefault('R5_STOP_AT_UNIX', 'inf')
    env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.9')
    subprocess.run([sys.executable, str(HERE / 'scripts' / script)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init')
    for key in ['run', 'openpi', 'checkpoint', 'data']:
        init.add_argument('--' + key, required=True)
    init.add_argument('--method', choices=['baseline', 'sma'], required=True)
    init.add_argument('--seed', type=int, default=42)
    for name in ['prepare', 'train']:
        commands.add_parser(name).add_argument('--run', required=True)
    args = parser.parse_args()
    initialize(args) if args.command == 'init' else launch(args)


if __name__ == '__main__':
    main()
