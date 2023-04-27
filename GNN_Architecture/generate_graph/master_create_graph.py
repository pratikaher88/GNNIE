
import subprocess

# 1. created graph with alll features
# # needs unified CSV
# subprocess.run(["python3", "generate_graph/feature_engineering.py"])

# # creates graph with engineered features
# subprocess.run(["python3", "generate_graph/modified_create_graph_using_features.py"])

# 2. create graphs without only two features
# do not remove duplicates
# subprocess.run(["python3", "generate_graph/create_and_save_graph.py"])
# remove duplicates
# subprocess.run(["python3", "generate_graph/create_and_save_graph.py", '--remove_duplicates'])

# 3. add structural features to graph
subprocess.run(["python3", "generate_graph/add_structural_features.py", '--graph_name', "ecommerce_hetero_graph" ])

