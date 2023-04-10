from collections import defaultdict


# validate if customer features in the graph match those in the dataframe
def validate_customer_features(clean_graph, df):
    cust_features = clean_graph.nodes['customer'].data['features'].tolist()
    cust_ids_from_graph_tensor = clean_graph.nodes('customer')
    cust_ids_from_graph = [id.item() for id in cust_ids_from_graph_tensor]

    print("----------- validating customer features ---------")

    id_feat_map = {}
    for cust_id, cust_feat in zip(cust_ids_from_graph, cust_features):
        id_feat_map[cust_id] = cust_feat

    validated_customer_ids = set()
    is_mismatch = False
    for index, row in df.iterrows():
        if row['customer_id_int'] in id_feat_map:
            if id_feat_map[row['customer_id_int']] == [float(row['customer_zip_code_prefix'])]:
                validated_customer_ids.add(row['customer_id_int'])
            else:
                print(f"mismatch in customer ID: {row['customer_id_int']}")
                is_mismatch = True

    true_count = len(validated_customer_ids)

    # check if all customer IDs present in the graph have been validated
    if true_count == clean_graph.num_nodes('customer') and is_mismatch == False:
        return "validate_customer_features: features match!"
    else:
        return "validate_customer_features: features mis-match."


# validate if product features in the graph match those in the dataframe
def validate_product_features(clean_graph, df):
    prod_features = clean_graph.nodes['product'].data['features'].tolist()
    prod_ids_from_graph_tensor = clean_graph.nodes('product')
    prod_ids_from_graph = [id.item() for id in prod_ids_from_graph_tensor]

    print("----------- validating product features ---------")

    id_feat_map = {}
    for prod_id, prod_feat in zip(prod_ids_from_graph, prod_features):
        id_feat_map[prod_id] = prod_feat

    validated_product_ids = set()
    is_mismatch = False
    for index, row in df.iterrows():
        if row['product_id_int'] in id_feat_map:
            graph_feat = id_feat_map[row['product_id_int']]
            if round(graph_feat[0], 2) == round(row['price'], 2) and graph_feat[1] == float(row['purchase_weekofyear']):
                validated_product_ids.add(row['product_id_int'])
            else:
                # please leave these commented lines here to aid future debugging
                # print(f"mismatch in product ID: {row['product_id_int']}")
                # print(f"prices: {round(graph_feat[0], 2)}, {round(row['price'], 2)}")
                # print(f"weekofyear: {graph_feat[1]}, {float(row['purchase_weekofyear'])}")
                if row['product_id_int'] not in validated_product_ids:
                    is_mismatch = True

    true_count = len(validated_product_ids)
    print(true_count)
    print(clean_graph.num_nodes('product'))

    # check if all product IDs present in the graph have been validated
    if true_count == clean_graph.num_nodes('product') and is_mismatch == False:
        return "validate_product_features: features match!"
    else:
        return "validate_product_features: features mis-match."


# check if edge assignment was done correctly
def validate_edges(clean_graph, df, etype, src_col_name, dest_col_name):
    src = [i.item() for i in clean_graph.edges(etype=etype)[0]]
    dest = [i.item() for i in clean_graph.edges(etype=etype)[1]]
    edges_dict = defaultdict(lambda: 0)

    # create a dictionary of form { (src id, dest id) = number of edges }
    for u, v in zip(src, dest):
        edges_dict[(u, v)] += 1

    # flag which is set to true if there is a mismatch
    is_mismatch = False

    # the below subtracts an edge every time it is found in the dataframe
    # if the count in the dict is < 0 or > 0, there is a mismatch in graph creation
    for index, row in df.iterrows():
        if (row[src_col_name], row[dest_col_name]) in edges_dict.keys():
            edges_dict[(row[src_col_name], row[dest_col_name])] -= 1

            # if there  is an extra edge in the dataframe  - scenario 1
            if edges_dict[(row[src_col_name], row[dest_col_name])] < 0:
                is_mismatch = True
                print("an edge in the dataset is not present in the graph! :(")
                break
        else:
            # if there  is an extra edge in the dataframe  - scenario 2
            print("an edge in the dataset is not present in the graph! :(")
            is_mismatch = True
    for val in edges_dict.values():
        if val != 0:
            is_mismatch = True

    if is_mismatch:
        return "There is a mismatch in edge assignment."
    return f"Edge assignment for etype {etype} is valid!"


# TODO: once we decide edge features, validate edge features
def validate_edge_features(clean_graph, df, etype, src_col_name, dest_col_name):
    prod_ids_from_graph_tensor = clean_graph.nodes('customer')
    prod_ids_from_graph = [id.item() for id in prod_ids_from_graph_tensor]
    edge_id_dict = {}
    edge_feat_dict = {}
    for node_id in prod_ids_from_graph:
        output = clean_graph.out_edges(node_id, etype='orders', form='all')
        print("u", output[0].item())
        print("v", output[1].item())
        print("edge ID", output[2].item())
        u = output[0].item()
        v = output[1].item()
        edge_id = output[2].item()
        edge_feat_dict[(u,v)] = edge_id

    edge_feats = clean_graph.edges['orders'].data['features'].tolist()
    for i in edge_feats:
        edge_feat_dict[i]
