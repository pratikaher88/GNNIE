import dgl, torch
import numpy as np
from model import ConvModel
from loss import max_margin_loss

np.random.seed(111)

graphs, _ = dgl.load_graphs("graph_files/ecommerce_hetero_graph.dgl")
ecommerce_hetero_graph = graphs[0]

# subgraph

ecommerce_hetero_graph_subgraph = ecommerce_hetero_graph.subgraph({ 'customer' :list(range(1000)), 'product': list(range(ecommerce_hetero_graph.num_nodes('product')))})
eids = np.arange(ecommerce_hetero_graph.number_of_edges(etype='orders'))


dim_dict = {'customer': ecommerce_hetero_graph_subgraph.nodes['customer'].data['features'].shape[1],
            'product': ecommerce_hetero_graph_subgraph.nodes['product'].data['features'].shape[1],
            'edge_dim': ecommerce_hetero_graph_subgraph.edges['orders'].data['features'].shape[1],
            'hidden_dim' : 128,
            'out_dim': 64
           }

# Train test split

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

# dataloader

neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1])

edge_sampler = dgl.dataloading.EdgePredictionSampler(
    node_sampler,
    negative_sampler=neg_sampler,
    exclude='self')

dataloader = dgl.dataloading.DataLoader(ecommerce_hetero_graph_subgraph, train_eids_dict, 
                                            edge_sampler,  shuffle=True, 
                                            batch_size=1024, num_workers=0)

num_batches = len(dataloader)
print("Number of batches ",len(dataloader))


# print(train_g.edata['features'])
# save down graphs

# save down graphs
dgl.save_graphs("graph_files/train_g.dgl", [train_g])
dgl.save_graphs("graph_files/valid_g.dgl", [valid_g])
dgl.save_graphs("graph_files/test_g.dgl", [test_g])
dgl.save_graphs("graph_files/ecommerce_hetero_graph_subgraph.dgl", [ecommerce_hetero_graph_subgraph])

# model building


model = ConvModel(ecommerce_hetero_graph_subgraph, 3, dim_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=0)

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
        
    #     # print("Edge Features shape : ", HM['orders'][0].shape, HM['rev-orders'][0].shape)
    #     # print(input_features)
    #     # print(len(blocks))

        _, pos_score, neg_score = model(blocks, input_features, HM, pos_g, neg_g)

        loss = max_margin_loss(pos_score, neg_score)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch += 1

        # print(f'batch: {batch} of {num_batches}')
    
    print(f'Total loss at epoch {i} :',total_loss)

    # # torch.save(model, 'mpnn_model_save.pth')
    # # torch.save(model.state_dict(), 'graph_files/trained_model.pth')
    # torch.save({'epoch': i,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': total_loss}, 
	#         'graph_files/trained_model.pth')