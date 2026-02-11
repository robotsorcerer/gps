#!/usr/bin/env python3
"""
Validation script to compare baseline vs modernized outputs.

This script runs experiments in both Python 2.7 and Python 3.11
environments and validates numerical equivalence.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentValidator:
    """Validates experiments between baseline and modernized implementations."""

    def __init__(
        self,
        tolerance: float = 1e-6,
        baseline_container: str = "gps-baseline",
        modern_container: str = "gps-modernized"
    ):
        self.tolerance = tolerance
        self.baseline_container = baseline_container
        self.modern_container = modern_container

    def run_baseline(self, experiment_dir: str) -> Dict[str, Any]:
        """Run experiment in Python 2.7 baseline container."""
        logger.info(f"Running baseline for {experiment_dir}")

        cmd = [
            "docker", "exec", self.baseline_container,
            "python2", "/gps/python/gps/gps_main.py", experiment_dir
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=True
            )
            logger.info("Baseline execution successful")
            return self._parse_output(result.stdout, experiment_dir)
        except subprocess.TimeoutExpired:
            logger.error(f"Baseline timed out for {experiment_dir}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Baseline failed: {e.stderr}")
            raise

    def run_modernized(self, experiment_dir: str) -> Dict[str, Any]:
        """Run experiment in Python 3.11 modernized container."""
        logger.info(f"Running modernized for {experiment_dir}")

        cmd = [
            "docker", "exec", self.modern_container,
            "python3", "/catkin_ws/src/gps/python/gps/gps_main.py", experiment_dir
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                check=True
            )
            logger.info("Modernized execution successful")
            return self._parse_output(result.stdout, experiment_dir)
        except subprocess.TimeoutExpired:
            logger.error(f"Modernized timed out for {experiment_dir}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Modernized failed: {e.stderr}")
            raise

    def _parse_output(self, stdout: str, experiment_dir: str) -> Dict[str, Any]:
        """Parse experiment output and extract key metrics."""
        # Load saved data from experiment directory
        data_dir = Path(experiment_dir) / "data_files"

        results = {}

        # Load costs
        costs_file = data_dir / "costs.csv"
        if costs_file.exists():
            results['costs'] = np.loadtxt(costs_file, delimiter=',')

        # Load policy parameters
        policy_file = data_dir / "policy_params.pkl"
        if policy_file.exists():
            import pickle
            with open(policy_file, 'rb') as f:
                results['policy'] = pickle.load(f)

        # Load trajectories
        traj_files = list(data_dir.glob("traj_*.pkl"))
        if traj_files:
            import pickle
            trajectories = []
            for traj_file in sorted(traj_files):
                with open(traj_file, 'rb') as f:
                    trajectories.append(pickle.load(f))
            results['trajectories'] = trajectories

        return results

    def validate_equivalence(
        self,
        baseline: Dict[str, Any],
        modernized: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate numerical equivalence between baseline and modernized.

        Returns:
            (is_valid, metrics_dict) where metrics_dict contains max differences
        """
        is_valid = True
        metrics = {}

        for key in baseline.keys():
            if key not in modernized:
                logger.error(f"Key '{key}' missing in modernized output")
                is_valid = False
                continue

            baseline_val = baseline[key]
            modern_val = modernized[key]

            # Handle numpy arrays
            if isinstance(baseline_val, np.ndarray):
                if not isinstance(modern_val, np.ndarray):
                    logger.error(f"Type mismatch for '{key}'")
                    is_valid = False
                    continue

                if baseline_val.shape != modern_val.shape:
                    logger.error(
                        f"Shape mismatch for '{key}': "
                        f"{baseline_val.shape} vs {modern_val.shape}"
                    )
                    is_valid = False
                    continue

                diff = np.abs(baseline_val - modern_val)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                metrics[f"{key}_max_diff"] = max_diff
                metrics[f"{key}_mean_diff"] = mean_diff

                if max_diff > self.tolerance:
                    logger.error(
                        f"'{key}' differs by {max_diff:.2e} "
                        f"(tolerance: {self.tolerance:.2e})"
                    )
                    is_valid = False
                else:
                    logger.info(
                        f"'{key}' validated: max_diff={max_diff:.2e}, "
                        f"mean_diff={mean_diff:.2e}"
                    )

            # Handle lists of arrays (trajectories)
            elif isinstance(baseline_val, list):
                if len(baseline_val) != len(modern_val):
                    logger.error(f"Length mismatch for '{key}'")
                    is_valid = False
                    continue

                for i, (base_item, mod_item) in enumerate(zip(baseline_val, modern_val)):
                    if isinstance(base_item, np.ndarray):
                        diff = np.abs(base_item - mod_item)
                        max_diff = np.max(diff)

                        if max_diff > self.tolerance:
                            logger.error(
                                f"'{key}[{i}]' differs by {max_diff:.2e}"
                            )
                            is_valid = False

        return is_valid, metrics

    def validate_experiment(self, experiment_dir: str) -> bool:
        """Run full validation pipeline for an experiment."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating experiment: {experiment_dir}")
        logger.info(f"{'='*60}\n")

        try:
            # Run baseline
            baseline = self.run_baseline(experiment_dir)

            # Run modernized
            modernized = self.run_modernized(experiment_dir)

            # Validate
            is_valid, metrics = self.validate_equivalence(baseline, modernized)

            # Save validation report
            report = {
                'experiment': experiment_dir,
                'valid': is_valid,
                'metrics': metrics,
                'tolerance': self.tolerance
            }

            report_file = Path(experiment_dir) / "validation_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"\nValidation report saved to {report_file}")

            if is_valid:
                logger.info(f"✅ PASS: {experiment_dir}")
            else:
                logger.error(f"❌ FAIL: {experiment_dir}")

            return is_valid

        except Exception as e:
            logger.exception(f"Validation failed with exception: {e}")
            return False


def get_all_experiments(base_dir: str = "experiments") -> list:
    """Get list of all experiment directories."""
    base_path = Path(base_dir)
    experiments = [
        str(d) for d in base_path.iterdir()
        if d.is_dir() and (d / "hyperparams.py").exists()
    ]
    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description="Validate GPS experiments between baseline and modernized versions"
    )
    parser.add_argument(
        '--experiment',
        type=str,
        help='Single experiment directory to validate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all experiments'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-6,
        help='Numerical tolerance for validation (default: 1e-6)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel validations (default: 1)'
    )

    args = parser.parse_args()

    validator = ExperimentValidator(tolerance=args.tolerance)

    if args.all:
        experiments = get_all_experiments()
        logger.info(f"Found {len(experiments)} experiments to validate")

        results = {}
        for exp in experiments:
            results[exp] = validator.validate_experiment(exp)

        # Summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)

        logger.info(f"\n{'='*60}")
        logger.info(f"VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success rate: {100 * passed / total:.1f}%")

        sys.exit(0 if passed == total else 1)

    elif args.experiment:
        success = validator.validate_experiment(args.experiment)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
