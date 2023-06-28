import subprocess

# subprocess.run(['python3', 'train_subgraph.py'])

subprocess.run(['python3', 'analysis_scripts/generate_trained_embeddings.py'])

subprocess.run(['python3', 'analysis_scripts/evaluate_trained_embeddings.py'])
