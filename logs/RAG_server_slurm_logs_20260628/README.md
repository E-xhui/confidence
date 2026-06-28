# RAG Server Slurm Logs Package

This package organizes the server-side Slurm logs copied from `gpu01:/data/home/yixh/RAG` plus a local log snapshot from `/Users/yixuhui/Desktop/RAG`.

Start with:

1. `metadata/job_id_index.md` for the human-readable job id summary.
2. `metadata/job_id_index.csv` if you want to filter/sort by job id or run.
3. `metadata/log_file_manifest.csv` for every copied log file with size, line count, timestamp, and last non-empty line.

Collection notes:

- Server source: `gpu01` / hostname `gpu-01`, user `yixh`, path `/data/home/yixh/RAG`.
- The `login` SSH alias timed out during this run, while `gpu01` was reachable.
- Slurm accounting commands were not available on `gpu01` PATH, so job ids were recovered from the Slurm log filenames and sbatch templates.
- Empty `.out` or `.err` files are preserved; many array jobs write warnings to `.err` while `.out` remains empty.
