"""
run_pipeline.py
---------------
Convenience orchestrator for the Modern Data Stack Supply Chain project.
Can trigger data generation, dbt parsing/running, and launch Streamlit.
"""

import os
import sys
import subprocess
import argparse

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DBT_DIR = os.path.join(WORKSPACE_DIR, "supply_chain_analytics")


def run_command(cmd, cwd=WORKSPACE_DIR):
    print(f"\n[*] Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"[!] Command failed with exit code: {res.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Modern Data Stack Orchestration CLI")
    parser.add_argument("--step", choices=["all", "docker", "data", "dbt", "test", "dashboard"], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("  SUPPLY CHAIN ANALYTICS: PIPELINE ORCHESTRATOR")
    print("=" * 70)

    if args.step in ["all", "docker"]:
        print("\n[Step 1] Spinning up PostgreSQL via Docker Compose...")
        run_command("docker compose up -d")

    if args.step in ["all", "data"]:
        print("\n[Step 2] Generating 100k messy supply chain records...")
        run_command(f"{sys.executable} generate_data.py")

    if args.step in ["all", "dbt"]:
        print("\n[Step 3] Running dbt models (src -> stg -> marts)...")
        run_command(
            f"{sys.executable} -c \"from dbt.cli.main import dbtRunner; dbtRunner().invoke(['run', '--project-dir', 'supply_chain_analytics', '--profiles-dir', 'supply_chain_analytics'])\""
        )

    if args.step in ["all", "test"]:
        print("\n[Step 4] Executing dbt data quality test suite...")
        run_command(
            f"{sys.executable} -c \"from dbt.cli.main import dbtRunner; dbtRunner().invoke(['test', '--project-dir', 'supply_chain_analytics', '--profiles-dir', 'supply_chain_analytics'])\""
        )

    if args.step in ["all", "dashboard"]:
        print("\n[Step 5] Launching Streamlit Monte Carlo Dashboard...")
        print("[*] Command to launch: streamlit run streamlit_app.py")
        run_command(f"{sys.executable} -m streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
