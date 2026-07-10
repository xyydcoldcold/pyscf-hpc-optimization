import argparse
import json
import os
import platform
import time
from pathlib import Path

import pyscf
from pyscf import gto, scf
from pyscf.tools import fcidump


# Root directory of the project, resolved from this benchmark file location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Directory for molecule/input data files.
DATA_DIR = PROJECT_ROOT / "data"
# Directory for benchmark result artifacts.
RESULTS_DIR = PROJECT_ROOT / "results"
# Directory containing benchmark scripts.
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

MOLECULE_FILES = {path.stem: path for path in sorted(DATA_DIR.glob("*.xyz"))}
DEFAULT_MOLECULE = "h2" if "h2" in MOLECULE_FILES else next(iter(MOLECULE_FILES), None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE, choices=MOLECULE_FILES.keys())
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--output-dir", default=RESULTS_DIR.relative_to(PROJECT_ROOT))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--export-fcidump", action="store_true")
    return parser.parse_args()


def read_xyz_geometry(path: Path) -> str:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Molecule file is empty: {path}")

    atom_lines = lines
    if lines[0].isdigit():
        atom_count = int(lines[0])
        atom_lines = lines[2 : 2 + atom_count]

    return "; ".join(atom_lines)


def build_molecule(geometry: str, basis: str):
    mol = gto.M(
        atom=geometry,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    return mol


def run_rhf(mol):
    mf = scf.RHF(mol)

    start = time.perf_counter()
    energy = mf.kernel()
    end = time.perf_counter()

    return mf, {
        "converged": bool(mf.converged),
        "e_tot": float(energy),
        "wall_time_sec": end - start,
        "nao": int(mol.nao_nr()),
        "nelectron": int(mol.nelectron),
        "nbas": int(mol.nbas),
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = read_xyz_geometry(MOLECULE_FILES[args.molecule])
    mol = build_molecule(geometry, args.basis)
    mf, result = run_rhf(mol)

    metadata = {
        "input": {
            "molecule": args.molecule,
            "geometry": geometry,
            "basis": args.basis,
            "threads": args.threads,
        },
        "result": result,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyscf_version": pyscf.__version__,
        },
    }

    json_path = output_dir / f"{args.molecule}_{args.basis}_rhf.json"
    save_json(metadata, json_path)

    if args.export_fcidump:
        fcidump_path = output_dir / f"{args.molecule}_{args.basis}.fcidump"
        fcidump.from_scf(mf, str(fcidump_path))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
