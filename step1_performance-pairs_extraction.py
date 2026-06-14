# Extract architecture-performance pairs from synetune
# 5^6 = 15625 configurations
# 6 hps: [hp_x0	hp_x1 hp_x2	hp_x3 hp_x4	hp_x5]
# 7 performances: ['metric_valid_error', 'metric_train_error', 'metric_runtime', 'metric_elapsed_time', 'metric_latency', 'metric_flops', 'metric_params']
# 1, 12, 36, 200 epochs

import itertools

import pandas as pd
from syne_tune.blackbox_repository import load_blackbox

from paths import DATA_DIR, NASBENCH_CSV

# Step 1: Load NAS-Bench-201 blackbox (choose one dataset)
# e.g. "cifar10", "cifar100", "ImageNet16-120"
blackbox = load_blackbox("nasbench201")["cifar10"]

# configuration space, architecture descriptions
config_space = blackbox.configuration_space

# Each hp_xi has 5 possible operations (categorical)
hp_names = list(config_space.keys())
hp_domains = [config_space[hp].categories for hp in hp_names]

print("Hyperparameters:", hp_names)
print("Each HP domain size:", [len(d) for d in hp_domains])

# Step 2: Generate all 15,625 architectures
# Cartesian product of all choices → full search space
all_configs = list(itertools.product(*hp_domains))
print(f"Total architectures: {len(all_configs)}")  # should be 15625

# Step 3: Query blackbox for performance at epoch=200
results = []

for cfg_tuple in all_configs:
    # Convert tuple to config dict: {"hp_x0": "...", ..., "hp_x5": "..."}
    config = {hp_names[i]: cfg_tuple[i] for i in range(len(hp_names))}

    # Query the blackbox at final fidelity (epoch 200)
    perf = blackbox.objective_function(
        configuration=config,
        fidelity={"hp_epoch": 1},
    )

    # Combine config and performance into one record
    row = {**config, **perf}
    results.append(row)

# Step 4: Save as CSV
df = pd.DataFrame(results)
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(NASBENCH_CSV, index=False)

print("Saved CSV with shape:", df.shape)
print(f"File: {NASBENCH_CSV}")
