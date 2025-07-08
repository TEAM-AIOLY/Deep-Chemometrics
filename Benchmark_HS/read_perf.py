import os
import pandas as pd

base_path ="C:/00_aioly/GitHub/Deep-Chemometrics/Benchmark_HS/mango/"

def read_metrics(metrics_file):
    metrics_dict = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                metrics_dict[key] = value
    return metrics_dict

metrics_per_model = {}

for model_name in os.listdir(base_path):
    model_path = os.path.join(base_path, model_name)
    if not os.path.isdir(model_path):
        continue
    runs_metrics = []
    for run_name in os.listdir(model_path):
        run_path = os.path.join(model_path, run_name)
        metrics_file = os.path.join(run_path, 'metrics.txt')
        if os.path.isdir(run_path) and os.path.isfile(metrics_file):
            metrics = read_metrics(metrics_file)
            metrics['Run'] = run_name
            runs_metrics.append(metrics)
    if runs_metrics:
        metrics_per_model[model_name] = runs_metrics


best_runs_rows = []


for model, runs in metrics_per_model.items():
    # Check which metric exists
    if 'R2' in runs[0]:
        key = 'R2'
        best_run = max(runs, key=lambda d: d.get('R2', float('-inf')))
        best_runs_rows.append({
            'Model': model,
            'Run': best_run.get('Run'),
            'R2': best_run.get('R2'),
            'RMSEP': best_run.get('RMSEP', best_run.get('RMSE')),  # handle both RMSEP and RMSE
            'N_parameters': best_run.get('N_parameters')
        })
    elif 'F1' in runs[0]:
        key = 'F1'
        best_run = max(runs, key=lambda d: d.get('F1', float('-inf')))
        best_runs_rows.append({
            'Model': model,
            'Run': best_run.get('Run'),
            'F1': best_run.get('F1'),
            'accuracy': best_run.get('accuracy'),
            'mean_precision': best_run.get('mean_precision'),
            'mean_recall': best_run.get('mean_recall'),
            'N_parameters': best_run.get('N_parameters')
        })

# Create DataFrame
df_best = pd.DataFrame(best_runs_rows)
print(df_best)




# all_metrics = []

# # List all entries in the base path, filter directories
# for folder_name in os.listdir(base_path):
#     folder_path = os.path.join(base_path, folder_name)
#     if os.path.isdir(folder_path):
#         metrics_file = os.path.join(folder_path, 'metrics.txt')
#         if os.path.isfile(metrics_file):
#             metrics_dict = {}
#             with open(metrics_file, 'r') as f:
#                 for line in f:
#                     if ':' in line:
#                         key, value = line.split(':', 1)
#                         key = key.strip()
#                         value = value.strip()
#                         try:
#                             if '.' in value:
#                                 value = float(value)
#                             else:
#                                 value = int(value)
#                         except ValueError:
#                             pass
#                         metrics_dict[key] = value
        
#             all_metrics.append(metrics_dict)
            
            
# runs = []
# rmsep = []
# r2 = []
# n_params = []
# best_epoch = []

# for i, metrics in enumerate(all_metrics, start=1):  # start=1 for 1-based run numbering
#     runs.append(f'Run {i}')
#     rmsep.append(metrics.get('RMSE', None))
#     r2.append(metrics.get('R2', None))
#     n_params.append(metrics.get('N_parameters', None))
#     best_epoch.append(metrics.get('best_epoch', None))

# df = pd.DataFrame({
#     'Run': runs,
#     'RMSE': rmsep,
#     'R2': r2,
#     'N_parameters': n_params,
#     'Best epoch': best_epoch,
# })

# print(df)