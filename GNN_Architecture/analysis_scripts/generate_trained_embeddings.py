import torch, dgl, os, yaml
from settings import BASE_DIR, CONFIG_PATH
from Model.model import ConvModel

saved_model = torch.load(f"{BASE_DIR}/graph_files_subgraph/trained_model.pth")

def load_config(config_name):
    with open(os.path.join(f"{CONFIG_PATH}", config_name)) as file:
        config = yaml.safe_load(file)
    return config

print("Loading model config")
model_config = load_config("model_config.yml")

graph_name = model_config['input_graph_name']
graph_details = model_config['graph_details']
graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/{graph_name}")
ecommerce_hetero_graph_subgraph = graphs[0]

print("Ecommerce graph", ecommerce_hetero_graph_subgraph)

# ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph_subgraph, { 'orders' : list(range(number_of_egdes)), 'rev-orders' : list(range(number_of_egdes)) } )
# ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph, { 'orders' : [random.randint(1, 10000) for i in range(1000)], 'rev-orders' : [random.randint(1, 10000) for i in range(1000)] } )

if model_config['train_full'] == False:
    number_of_egdes = model_config['number_of_egdes']
    ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph_subgraph, { 'orders' : list(range(number_of_egdes)), 'rev-orders' : list(range(number_of_egdes)) } )
    # ecommerce_hetero_graph_subgraph = dgl.edge_subgraph(ecommerce_hetero_graph, { 'orders' : [random.randint(1, 10000) for i in range(1000)], 'rev-orders' : [random.randint(1, 10000) for i in range(1000)] } )



neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

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
            'out_dim': model_config['output_dim']}

graphs, _ = dgl.load_graphs(f"{BASE_DIR}/graph_files_subgraph/valid_g.dgl")
valid_g = graphs[0]


mpnn_model = ConvModel(ecommerce_hetero_graph_subgraph, model_config['num_layers'], dim_dict, aggregator_type=model_config['aggregate_fn'], pred=model_config['pred'])
mpnn_model.load_state_dict(saved_model['model_state_dict'])
mpnn_model.eval()

# print(model_config)

print(f"Validating model : {valid_g}",)

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
# node_sampler = dgl.dataloading.NeighborSampler(fanouts=[1, 1])
node_sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in model_config['fanouts'].split(',') ], replace=False)

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

valid_eids_dict = {}
for e in valid_g.edata[dgl.EID].keys():
    valid_eids_dict[e[1]] = valid_g.edata[dgl.EID][e]

valid_dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, valid_eids_dict, edge_sampler,  shuffle=True, batch_size=model_config['batch_size'], num_workers=model_config['num_workers'])

print(valid_eids_dict['orders'].shape,ecommerce_hetero_graph_subgraph.num_edges('orders'))
# print(next(iter(valid_dataloader)))


train_embeddings = {ntype: torch.zeros(valid_g.num_nodes(ntype), dim_dict['out_dim'], requires_grad=False)
         for ntype in valid_g.ntypes}

batch, num_batches = 0, len(valid_dataloader)

for _ , pos_g, neg_g, blocks in valid_dataloader:

    output_nodes = pos_g.ndata[dgl.NID]

    input_features = blocks[0].srcdata['features']
    edge_features = blocks[0].edata['features']

    edge_features_HM = {}
    for key, value in edge_features.items():
        # print(value.grad)
        edge_features_HM[key[1]] = (value.detach(), )
        # edge_features_HM[key[1]] = value.detach()

    # print(edge_features_HM['rev-orders'], input_features['customer'].shape, input_features['product'].shape)
    
    input_features['customer'] = mpnn_model.user_embed(input_features['customer'].detach()).squeeze().detach()
    input_features['product'] = mpnn_model.item_embed(input_features['product'].detach()).squeeze().detach()

    # print(blocks, len(blocks), "Input features shape", input_features['customer'].grad, input_features['product'].grad)

    # print("Input features shape", input_features, input_features['customer'].shape, input_features['product'].shape)
    # print("Edge features", edge_features_HM['orders'][0].shape, edge_features_HM['rev-orders'][0].shape)
    
    h = mpnn_model.get_repr(blocks, input_features)
    # h = mpnn_model.get_repr(blocks, input_features, edge_features_HM)

    h['customer'] = h['customer'].detach()
    h['product'] = h['product'].detach()

    for ntype in h.keys():
        train_embeddings[ntype][output_nodes[ntype]] = h[ntype].detach()

    batch += 1
    
    print(f'batch: {batch} of {num_batches}')


import pickle
with open( f'{BASE_DIR}/graph_files_subgraph/trained_embeddings.pickle', 'wb') as f:
    pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
