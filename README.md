TPUv4 Resiliency Simulator
This project contains a small Monte Carlo simulator inspired by Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer by Zu et al. The simulator studies why large, tightly coupled jobs become difficult to schedule when resources fail or are occupied by other workloads.
What is modeled?
The simulator treats each TPUv4 cube as one allocation unit. A cube can be free, failed, or occupied.
- Static allocation: a job must fit inside one fixed contiguous group of cubes.
- Reconfigurable allocation: a job can use any healthy, free cubes because the logical network can be reconfigured.
This is an abstraction, not a TPU emulator. It does not require a cloud account, GPU, TPU, or distributed cluster.
Ubuntu instructions
sudo apt update
sudo apt install -y python3 python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tpu_resiliency_sim.py --outdir results
The default run uses 5,000 trials per data point and should finish on a normal laptop. To run a smaller test:
python tpu_resiliency_sim.py --outdir results --trials 1000
Output
The program creates:
- experiment1_job_size.png and .csv: success rate as job size increases;
- experiment2_failures.png and .csv: success rate as failure probability increases;
- experiment3_fragmentation.png and .csv: effect of workload placement and fragmentation;
- results.json: all raw results.
The random seed is fixed by default so the graphs are reproducible. Use --seed to run a different sample.
Suggested GitHub contents
Commit tpu_resiliency_sim.py, README.md, requirements.txt, and the generated results/ directory. The report is submitted separately through Canvas. Before submission, replace the GitHub placeholder in the report with the actual private repository URL and confirm that the instructor has read access.
Connection to the paper
The model reflects the paper's discussion of static TPUv3 pods, TPUv4 cube-level configurability, workload fragmentation, and resource failures. The simulator intentionally leaves out physical ICI routing, OCS hardware, collective communication, correlated failures, recovery time, and scheduler priorities. Its purpose is to make the availability trend understandable with a laptop-scale experiment.
