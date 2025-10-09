import os
import itertools

base_path  ="C:/00_aioly/GitHub/Deep-chemometrics/Jean-zay_bench/wheat/"

results = []

for root, dirs, files in os.walk(base_path):
    if "metrics.txt" in files:
        metrics_file = os.path.join(root, "metrics.txt")
        with open(metrics_file, "r") as f:
            lines = f.readlines()
        metrics_dict = {}
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val or "e" in val.lower():
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                metrics_dict[key] = val
        
        sort_metric = None
        if "F1" in metrics_dict:
            sort_metric = metrics_dict["F1"]
        elif "RMSE" in metrics_dict:
            sort_metric = metrics_dict["RMSE"]

        if sort_metric is not None:
            folder_name = os.path.basename(root)
            results.append({"folder": folder_name, "sort_metric": sort_metric, **metrics_dict})

# Sort results: descending for F1, ascending for RMSEP/RMSE
if results:
    if "F1" in results[0]:
        results.sort(key=lambda x: x["sort_metric"], reverse=True)  # highest F1 first
    else:
        results.sort(key=lambda x: x["sort_metric"])  # lowest RMSEP first

# Print top 3 results
print("\nTop 3 runs:")
for i, res in enumerate(results[:3], 1):
    print(f"\nRank {i}:")
    for k, v in res.items():
        print(f"  {k}: {v}")    
            

# # --- Parameter grid ---
# base_params = {
#     "spec_dims": None,
#     "mean": None,
#     "std": None,
#     "DP": 0.1,
#     "LR": 0.001,
#     "EPOCH": 2000,
#     "WD": 0.0055,
#     "batch_size": 512
# }


# LR  = [0.0001,0.00001]
# DP=[0.5,0.2,0.1]
# IP=[4,8]


# param_variations = [
#     {"LR": lr, "DP": dp, "IP": ip}
#     for lr, dp, ip in itertools.product(LR, DP, IP)
# ]
# paramsets = [{**base_params, **variation} for variation in param_variations]

# # Attach run_id to paramsets
# for i, param in enumerate(paramsets):
#     param["run_id"] = f"Run_{i:02d}"

# param_lookup = {p["run_id"]: p for p in paramsets}

# # --- Sort by metric ---
# if "wheat" in base_path.lower():
#     # Sort by F1 descending (max first)
#     results_sorted = sorted(results, key=lambda x: x["sort_metric"], reverse=True)
# else:
#     # Sort by RMSE ascending (min first)
#     results_sorted = sorted(results, key=lambda x: x["sort_metric"])

# top3 = results_sorted[:3]

# print("\n=== Top 3 results with parameters and all metrics ===\n")
# for res in top3:
#     run_id = res["folder"]  # assuming folder = Run_XX
#     print(f"{run_id}: sort_metric = {res['sort_metric']}")

#     print(" Metrics:")
#     for metric_key, metric_val in res.items():
#         if metric_key not in ("folder", "sort_metric"):
#             print(f"  {metric_key}: {metric_val}")

#     if run_id in param_lookup:
#         print(" Parameters:")
#         params = param_lookup[run_id]
#         for k, v in params.items():
#             if k != "run_id":
#                 print(f"  {k}: {v}")
#     else:
#         print("  (No matching params found!)")

#     print("-" * 50)
