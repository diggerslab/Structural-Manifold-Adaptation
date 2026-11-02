"""Fetch a pinned upstream checkout and apply only the detached-score hook."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def setup(destination):
    pin = json.loads((HERE / 'dependencies.json').read_text())['openpi']
    destination = Path(destination).resolve()
    if not destination.exists():
        subprocess.run(['git', 'clone', '--no-checkout', pin['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', pin['commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['commit']:
        raise ValueError('Use a dedicated checkout at the pinned commit; existing checkouts are not reset.')
    patch = HERE / 'patches/openpi-real7-score.patch'
    command = ['git', '-C', str(destination), 'apply']
    if subprocess.run(command + ['--reverse', '--check', str(patch)], capture_output=True).returncode == 0:
        return
    subprocess.run(command + ['--check', str(patch)], check=True)
    subprocess.run(command + [str(patch)], check=True)
    path = destination / 'src/openpi/models/pi0.py'
    print(json.dumps({'commit': actual, 'loss_hook_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    setup(parser.parse_args().destination)
