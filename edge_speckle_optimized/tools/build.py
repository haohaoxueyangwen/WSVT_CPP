"""Configure/build local native libraries and record the actual toolchain."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cmake', default='cmake')
    p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--arch', default='87', choices=['87','89'])
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()
    if args.jobs < 1:
        raise ValueError('Positive build jobs required')
    build = ROOT / 'build'
    configure = [args.cmake, '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_COMPILER=' + args.nvcc, '-DEDGE_CUDA_ARCH=' + args.arch,
                 '-DCMAKE_CUDA_ARCHITECTURES=' + args.arch]
    subprocess.run(configure, check=True)
    subprocess.run([args.cmake, '--build', str(build), '--target', 'edge_kernels', 'edge_spline', '-j', str(args.jobs)], check=True)
    def version(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    files = list((ROOT / 'native').glob('*')) + list((ROOT / 'speckle_edge/_native').glob('*.so')) + [ROOT / 'CMakeLists.txt']
    cache = (build / 'CMakeCache.txt').read_text()
    compilers = {}
    for line in cache.splitlines():
        if line.startswith(('CMAKE_CXX_COMPILER:FILEPATH=', 'CMAKE_CUDA_COMPILER:FILEPATH=')):
            key, value = line.split('=', 1)
            compilers[key.split(':')[0]] = value
    receipt = dict(schema_version=1, arch=args.arch, configure_command=configure,
                   cmake=version([args.cmake, '--version']), nvcc=version([args.nvcc, '--version']),
                   cxx=version([compilers['CMAKE_CXX_COMPILER'], '--version']),
                   cuda_compile_options=['-O3', '--std=c++17', '--fmad=false', '-arch=sm_' + args.arch],
                   cmake_cache=cache,
                   source_and_binary_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (ROOT / 'speckle_edge/_native/build_info.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Built and recorded native libraries in speckle_edge/_native')


if __name__ == '__main__':
    main()
