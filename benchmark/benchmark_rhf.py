import argparse
import json
import os
import platform
import resource
import statistics
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
NS_TO_SEC = 1_000_000_000
TIMING_PHASES = (
    "molecule_build_wall_sec",
    "scf_setup_wall_sec",
    "scf_kernel_wall_sec",
    "total_compute_wall_sec",
)


def elapsed_seconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / NS_TO_SEC


def peak_rss_mb() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak_rss / (1024 * 1024)
    return peak_rss / 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE, choices=MOLECULE_FILES.keys())
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--output-dir", default=RESULTS_DIR.relative_to(PROJECT_ROOT))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--export-fcidump", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


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


def run_single_rhf_measurement(geometry: str, basis: str):
    total_start_ns = time.perf_counter_ns()

    molecule_start_ns = total_start_ns
    mol = build_molecule(geometry, basis)
    molecule_end_ns = time.perf_counter_ns()

    scf_setup_start_ns = molecule_end_ns
    mf = scf.RHF(mol)
    scf_setup_end_ns = time.perf_counter_ns()

    scf_kernel_start_ns = scf_setup_end_ns
    energy = mf.kernel()
    scf_kernel_end_ns = time.perf_counter_ns()

    total_end_ns = scf_kernel_end_ns

    problem_size = {
        "natm": int(mol.natm),
        "nao": int(mol.nao_nr()),
        "nelectron": int(mol.nelectron),
        "nbas": int(mol.nbas),
    }

    timing = {
        "molecule_build_wall_sec": elapsed_seconds(
            molecule_start_ns, molecule_end_ns
        ),
        "scf_setup_wall_sec": elapsed_seconds(scf_setup_start_ns, scf_setup_end_ns),
        "scf_kernel_wall_sec": elapsed_seconds(
            scf_kernel_start_ns, scf_kernel_end_ns
        ),
        "total_compute_wall_sec": elapsed_seconds(total_start_ns, total_end_ns),
    }

    result = {
        "converged": bool(mf.converged),
        "cycles": int(mf.cycles),
        "e_tot": float(energy),
    }

    return mf, problem_size, timing, result


def summarize_timings(measurements):
    summary = {}
    for phase in TIMING_PHASES:
        values = [measurement["timing"][phase] for measurement in measurements]
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        summary[phase] = {
            "median": statistics.median(values),
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "cv_percent": std / mean * 100 if mean != 0 else 0.0,
        }
    return summary


def summarize_peak_rss(measurements):
    mean = statistics.mean(measurements)
    std = statistics.pstdev(measurements)
    return {
        "median_peak_rss_mb": statistics.median(measurements),
        "mean_peak_rss_mb": mean,
        "std_peak_rss_mb": std,
        "min_peak_rss_mb": min(measurements),
        "max_peak_rss_mb": max(measurements),
        "cv_percent": std / mean * 100 if mean != 0 else 0.0,
    }


def run_repeated_benchmark(geometry: str, basis: str, warmup: int, repeat: int):
    for _ in range(warmup):
        run_single_rhf_measurement(geometry, basis)

    measurements = []
    peak_rss_measurements = []
    for repetition in range(1, repeat + 1):
        mf, problem_size, run_timing, result = run_single_rhf_measurement(
            geometry, basis
        )
        peak_rss_measurements.append(peak_rss_mb())
        measurements.append(
            {
                "repetition": repetition,
                "timing": run_timing,
                "result": result,
            }
        )

    timing = {
        "clock": "perf_counter_ns",
        "unit": "seconds",
        "std_convention": "population",
        "warmup_count": warmup,
        "repeat_count": repeat,
        "measurements": measurements,
        "summary": summarize_timings(measurements),
    }
    memory = {
        "measurements": peak_rss_measurements,
        "summary": summarize_peak_rss(peak_rss_measurements),
    }
    return mf, problem_size, timing, memory


def runtime_parallel_configuration():
    return {
        "pyscf_threads": int(pyscf.lib.num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
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
    mf, problem_size, timing, memory = run_repeated_benchmark(
        geometry, args.basis, args.warmup, args.repeat
    )

    metadata = {
        "input": {
            "molecule": args.molecule,
            "geometry": geometry,
            "basis": args.basis,
            "threads": args.threads,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "problem_size": problem_size,
        "timing": timing,
        "memory": memory,
        "parallel": runtime_parallel_configuration(),
        "result": timing["measurements"][-1]["result"],
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
