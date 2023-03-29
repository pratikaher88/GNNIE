import dgl, torch
import numpy as np
from Model.model import ConvModel
from Model.loss import max_margin_loss
import time, yaml, os
from settings import BASE_DIR, MODEL_DIR, CONFIG_PATH

start = time.time()

np.random.seed(42)

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(f"{CONFIG_PATH}", config_name)) as file:
        config = yaml.safe_load(file)
    return config

model_config = load_config("model_config.yml")

# graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files/train_g.dgl")
# train_g = graphs[0]

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/{MODEL_DIR}/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph = graphs[0]

# eids = np.arange(ecommerce_hetero_graph.number_of_edges(etype='orders'))
# eids = np.random.permutation(eids)

# test_size = int(len(eids) * 0.1)
# valid_size = int(len(eids) * 0.1)
# train_size = len(eids) - test_size - valid_size

# train_eids_dict = {}
# for e in ecommerce_hetero_graph.etypes:
#     train_eids_dict[e] = eids[:train_size]

dim_dict = {'customer': ecommerce_hetero_graph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph.edges['orders'].data['features'].shape[1],
            'edge_hidden_dim': model_config['edge_hidden_dim'],
            'hidden_dim' : model_config['hidden_dim'],
            'out_dim': model_config['output_dim']
           }


# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)

# train_dataloader = dgl.dataloading.DataLoader(
#         train_g, train_eids_dict, 
#         sampler, 
#         # negative_sampler=dgl.dataloading.negative_sampler.Uniform(5), 
#         shuffle=True,
#         # drop_last=False,
#         batch_size = 32,
#         num_workers=0
#         )


sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

eids = np.arange(ecommerce_hetero_graph.number_of_edges(etype='orders'))
eids = np.random.permutation(eids)
train_eids_dict = {}
valid_eids_dict = {}
test_eids_dict = {}

test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size - valid_size

for e in ecommerce_hetero_graph.etypes:
    train_eids_dict[e] = eids[:train_size]
    valid_eids_dict[e] = eids[train_size:train_size+valid_size]
    test_eids_dict[e] = eids[-test_size:]

train_g = dgl.edge_subgraph(ecommerce_hetero_graph, train_eids_dict, relabel_nodes=False)
valid_g = dgl.edge_subgraph(ecommerce_hetero_graph, valid_eids_dict, relabel_nodes=False)
test_g = dgl.edge_subgraph(ecommerce_hetero_graph, test_eids_dict, relabel_nodes=False)

# Fix this : create a better subgraph
# smaller train graph
# train_g = ecommerce_hetero_graph.subgraph({ 'customer' :list(range(100)), 'product': list(range(train_g.num_nodes('product')))})
# eids = np.arange(train_g.number_of_edges(etype='orders'))

# train_eids_dict = {}
# for e in train_g.etypes:
#     train_eids_dict[e] = eids[:train_size]

# print(train_g)
# Fix this

# save down graphs
dgl.save_graphs(f"{BASE_DIR}/{MODEL_DIR}/train_g.dgl", [train_g])
dgl.save_graphs(f"{BASE_DIR}/{MODEL_DIR}/valid_g.dgl", [valid_g])
dgl.save_graphs(f"{BASE_DIR}/{MODEL_DIR}/test_g.dgl", [test_g])
dgl.save_graphs(f"{BASE_DIR}/{MODEL_DIR}/ecommerce_hetero_graph.dgl", [ecommerce_hetero_graph])

# dataloader = dgl.dataloading.DataLoader(
#     train_g, ids_dict, sampler,
#     batch_size=16,
#     shuffle=True,
#     drop_last=False,
#     num_workers=0)

# print(test_g.num_edges(etype='orders'), valid_g.num_edges(etype='orders'), train_g.num_edges(etype='orders'), ecommerce_hetero_graph.num_edges(etype='orders'))

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])
# node_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
# not sure which sampler to use when
edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    # reverse_etypes = {'orders' : 'rev-orders'},
    exclude='self')

# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

# dataloader = dgl.dataloading.DataLoader(train_g,    
#                                         train_eids_dict, 
#                                         sampler,    
#                                         # negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
#                                         shuffle=True, 
#                                         batch_size=16, 
#                                         num_workers=0)

# TODO : is it ecommerce_hetero_graph or train_g
dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph, train_eids_dict, 
                                            edge_sampler,  shuffle=True, 
                                            batch_size=model_config['batch_size'], num_workers=0)

num_batches = len(dataloader)
print("Number of batches ",len(dataloader))
# input_nodes, pos_g, neg_g, blocks = next(iter(dataloader))

# print(len(next(iter(dataloader))))

# print("Input Nodes", input_nodes ) 
# print("Pos G",pos_g)
# print("Neg G",neg_g)
# print("Blocks",blocks)

# print("reverse orders",train_g.edata)

model = ConvModel(ecommerce_hetero_graph, model_config['num_layers'], dim_dict, aggregator_type=model_config['aggregate_fn'])
optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'],weight_decay=0)


for i in range(10):

    total_loss = 0
    batch = 0

    for _, pos_g, neg_g, blocks in dataloader:

        optimizer.zero_grad()

        input_features = blocks[0].srcdata['features']

        edge_features = blocks[0].edata['features']

        HM = {}
        for key, value in edge_features.items():
            HM[key[1]] = (value, )
        
        # print("Edge Features shape : ", HM['orders'][0].shape, HM['rev-orders'][0].shape)

        # print(input_features)
        # print(len(blocks))

        _, pos_score, neg_score = model(blocks, input_features, HM, pos_g, neg_g)

        loss = max_margin_loss(pos_score, neg_score)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch += 1

        print(f'batch: {batch} of {num_batches}')

        # if batch % 10:
        #     print(f'batch: {batch} of {num_batches}')

        # _, pos_score, neg_score = model(blocks, input_nodes, pos_g, neg_g, input_nodes)
        # print(pos_score)
        # break
    
    print(f'Total loss at epoch {i} :',total_loss)
    # torch.save(model, 'mpnn_model_save.pth')
    # torch.save(model.state_dict(), 'graph_files/trained_model.pth')
    
    torch.save({'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss}, 
	        f'{BASE_DIR}/{MODEL_DIR}/trained_model.pth')


print(time.time() - start)
