# GNN Project 2 - Actor Node Classification

Run all commands from the project root.

Install dependencies: `pip install -r requirements.txt`

Run full model training and HPO:
`python src/run_milestone2.py --data_root data/actor --out_dir reports/milestone2 --seed 42 --num_trials 30`

Run depth-wise ablation:
`python src/analysis/ablations.py --data_root data/actor --best_params_csv reports/milestone2/tables/best_params.csv --out_dir reports/milestone2`

Run t-SNE plots: `python src/analysis/visualise_embeddings.py --data_root data/actor --best_params_csv reports/milestone2/tables/best_params.csv --out_dir reports/milestone2`

Run statistical tests: `python src/analysis/statistical_tests.py --results_json reports/milestone2/tables/all_results.json --per_class_csv reports/milestone2/tables/per_class_f1.csv`

Main outputs are saved under `reports/milestone2/figures`, `reports/milestone2/tables`, and `reports/milestone2/logs`.
