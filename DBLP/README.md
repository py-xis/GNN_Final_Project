# DBLP Project Folder

This folder contains the DBLP portion of the GNN project submission.

## Contents

- `DBLP_Node_Classification.ipynb`
  Main notebook for DBLP author node classification, benchmarking, ablation studies, metapath analysis, visualisation, and efficiency analysis.
- `requirements_dblp.txt`
  Python dependencies used for the DBLP notebook.
- `dblp_results_notebook.json`
  Saved main benchmark results.
- `dblp_ablation_results_notebook.json`
  Saved input and depth ablation results.
- `dblp_metapath_ablation_results_notebook.json`
  Saved metapath ablation results.
- `dblp_efficiency_results_notebook.json`
  Saved parameter count and training-time efficiency results.
- `data/`
  DBLP dataset files used by the notebook.
- `figures/`
  Exported t-SNE visualisation figures used in the report.
- `Utils/`
  Reference material and supporting PDFs/notebooks used during the project.

## Task

The DBLP task is author node classification. The goal is to predict the research area of each author node.

The evaluated models are:

- GCN
- GraphSAGE
- GAT
- HAN

## How To Run

1. Create a Python environment with the packages from `requirements_dblp.txt`.
2. Open `DBLP_Node_Classification.ipynb`.
3. Run the notebook cells in order.

The notebook uses the local `data/DBLP` directory and writes result JSON files into this folder.

## Notes

- The saved JSON result files are included so the main results can be inspected without rerunning every experiment.
- The efficiency results were generated in an appended notebook section and stored separately to avoid overwriting earlier benchmark outputs.
- All model training and inference for the DBLP experiments were performed in Google Colab using a T4 GPU runtime.
