import dgl, torch, pickle
import numpy as np
import os, random, yaml, time
from Model.model import ConvModel
from Model.loss import max_margin_loss, binary_cross_entropy_loss
from settings import BASE_DIR, CONFIG_PATH
from helper import get_model_size

start = time.time()

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(f"{CONFIG_PATH}", config_name)) as file:
        config = yaml.safe_load(file)
    return config

model_config = load_config("model_config.yml")

np.random.seed(42)

graph_name = model_config['input_graph_name']
graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/{graph_name}")
ecommerce_hetero_graph = graphs[0]

# subgraph
# ecommerce_hetero_graph_subgraph = ecommerce_hetero_graph.subgraph({ 'customer' :list(range(1000)), 'product': list(range(ecommerce_hetero_graph.num_nodes('product')))})
delta =  model_config['delta']

# check if the traijing is for subgraph for full graph
if model_config['train_full'] == True:
    print("Training on full graph")
    ecommerce_hetero_graph_subgraph = ecommerce_hetero_graph
else:
    number_of_egdes = model_config['number_of_egdes']
    ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph, { 'orders' : list(range(number_of_egdes)), 'rev-orders' : list(range(number_of_egdes)) } )
    # ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph, { 'orders' : [random.randint(1, 10000) for i in range(1000)], 'rev-orders' : [random.randint(1, 10000) for i in range(1000)] } )

print("Training Graph: ",ecommerce_hetero_graph_subgraph)
print("Input nodes shape : ", ecommerce_hetero_graph_subgraph.ndata['features']['customer'].shape, ecommerce_hetero_graph_subgraph.ndata['features']['product'].shape)

dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'orders' : {
                'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            },
            'rev-orders' : {
                'edge_dim': ecommerce_hetero_graph_subgraph.edges['rev-orders'].data['features'].shape[1],
            },
            'edge_hidden_dim': model_config['edge_hidden_dim'],
            'hidden_dim' : model_config['hidden_dim'],
            'out_dim': model_config['output_dim']
           }

# Divide into Train test split

eids = np.arange(ecommerce_hetero_graph_subgraph.number_of_edges(etype='orders'))
eids = np.random.permutation(eids)
train_eids_dict, valid_eids_dict, test_eids_dict = {}, {}, {}

test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = len(eids) - test_size - valid_size

for e in ecommerce_hetero_graph_subgraph.etypes:
    train_eids_dict[e] = eids[:train_size]
    valid_eids_dict[e] = eids[train_size:train_size+valid_size]
    test_eids_dict[e] = eids[-test_size:]

train_g = dgl.edge_subgraph(ecommerce_hetero_graph_subgraph, train_eids_dict, relabel_nodes=False)
valid_g = dgl.edge_subgraph(ecommerce_hetero_graph_subgraph, valid_eids_dict, relabel_nodes=False)
test_g = dgl.edge_subgraph(ecommerce_hetero_graph_subgraph, test_eids_dict, relabel_nodes=False)

# dataloader work

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
# node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])
# node_sampler = dgl.dataloading.NeighborSampler(fanouts=[1, 1])
node_sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in model_config['fanouts'].split(',') ], replace=False)

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, train_eids_dict, 
                                            edge_sampler,  shuffle=True, 
                                            batch_size=model_config['batch_size'], num_workers=model_config['num_workers'])

num_batches = len(dataloader)
print("Number of batches ",len(dataloader))

# print(train_g.edata['features'])
# save down graphs

# save down graphs
dgl.save_graphs(f"{BASE_DIR}/graph_files_subgraph/train_g.dgl", [train_g])
dgl.save_graphs(f"{BASE_DIR}/graph_files_subgraph/valid_g.dgl", [valid_g])
dgl.save_graphs(f"{BASE_DIR}/graph_files_subgraph/test_g.dgl", [test_g])
# uncomment this when running subgraphs
# dgl.save_graphs(f"{BASE_DIR}/graph_files_subgraph/{graph_name}", [ecommerce_hetero_graph_subgraph])

# with open( f'{BASE_DIR}/graph_files_subgraph/valid_eids_dict.pickle', 'wb') as f:
#     pickle.dump(valid_eids_dict, f, pickle.HIGHEST_PROTOCOL)

# model building

model = ConvModel(ecommerce_hetero_graph_subgraph, model_config['num_layers'], dim_dict, aggregator_type=model_config['aggregate_fn'], pred=model_config['pred'])

optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'],weight_decay=0)

print("Training begins")

for e in range(model_config['n_epochs']):

    total_loss = 0
    batch = 0

    print("Epoch :", e)

    for _, pos_g, neg_g, blocks in dataloader:

        optimizer.zero_grad()

        input_features = blocks[0].srcdata['features']
        edge_features = blocks[0].edata['features']

        edge_features_HM = {}
        for key, value in edge_features.items():
            edge_features_HM[key[1]] = (value, )
        
        # print("Edge Features shape : ", HM['orders'][0].shape, HM['rev-orders'][0].shape)
        # print(input_features['customer'].shape, input_features['product'].shape)
        # print(len(blocks))

        _, pos_score, neg_score = model(blocks, input_features, edge_features_HM, pos_g, neg_g)
        # _, pos_score, neg_score = model(blocks, input_features, edge_features, pos_g, neg_g)

        loss = max_margin_loss(pos_score, neg_score, delta)

        total_loss += loss.item()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch += 1

        # print(f'batch: {batch} of {num_batches}')
    
    print(f'Total loss at epoch {e} :',total_loss)
    print(f"Time taken so far : {(time.time() - start) / 60.0 :.2f}")

    if e!= 0 and e % 5 == 0:

        torch.save({'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss}, 
            f'{BASE_DIR}/graph_files_subgraph/trained_model.pth')

    # torch.save(model, 'mpnn_model_save.pth')
    # torch.save(model.state_dict(), 'f"{BASE_DIR}/graph_files/trained_model.pth')
get_model_size(model)

torch.save({'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss}, 
        f'{BASE_DIR}/graph_files_subgraph/trained_model.pth')

print("Training complete: saved model to :", f'{BASE_DIR}/graph_files_subgraph/trained_model.pth')


print(time.time() - start)