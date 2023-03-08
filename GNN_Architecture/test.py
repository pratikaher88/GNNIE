import torch, dgl, numpy as np
from settings import BASE_DIR

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_next_run/ecommerce_hetero_graph.dgl")
g = graphs[0]

print(g.in_degrees(etype='orders').sort())

exit(g.out_degrees(etype='orders').sort())

# g = dgl.rand_graph(300, 300)

valid_eids_dict = {}

# eids = np.arange(g.number_of_edges(etype='orders'))

# for e in g.etypes:
#     valid_eids_dict[e] = eids

import pickle 

with open( 'graph_files/valid_eids_dict.pickle', 'wb') as f:
    pickle.dump(valid_eids_dict, f, pickle.HIGHEST_PROTOCOL)

sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])

dataloader = dgl.dataloading.DataLoader(
    g, valid_eids_dict, sampler,
    batch_size=3, shuffle=True, drop_last=False, num_workers = 0
)

print(g)
print(valid_eids_dict)

input_nodes, output_nodes, blocks = next(iter(dataloader))

print(input_nodes, output_nodes)

# assert len(output_nodes) == batch_size
# assert blocks[-1].num_dst_nodes() == batch_size
# assert len(blocks) == 3