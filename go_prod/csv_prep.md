# csv_preprocessing module

Takes CSV as input, validate and clean it, create record-wise features and return a CSV.


### Input/Output

--- | Format | Description
 --- | --- | ---
 Input | CSV | Should be log files in CSV format. size=(N, F)
 Output | CSV | Ready for graph generataion. size=(N', F') where N'<= N and F'>=F


### Input params

Name | Type | Default | Description
--- | --- | --- | ---
user_node | str | - | user id in logs |
product_node | str | - | product id in logs|
user_features | list of str | - | user-related fields that users wish to include in the model, e.g. age, gender, membership status
product_features | list of str | - | product-related fields that users wish to include in the model, e.g. product category, price
edge_type | str | purchase | type of transactions in the log data, e.g. purchase/comment...
edge_features | list of str | - | transaction-specific fields that users wish to include in the model, e.g. transaction date, purchase amount, sale at discount
new_features | dict | - | specify the corresponding column for new feature generation. {new_feature1: [exist_feature1, ...]}


### CSV requirements

**(Would be best if implemented to UI)**
1. User id and product id are unique identifiers.
2. One CSV contains only one type of transaction (e.g. purchase, view, and like counts for 3 types).


### Functionality
1. validate user-upload data.
    - key columns (user, product) do exist
    - warn users if more than 20% of recirds are duplicated rows
2. data cleaning and manipulation
    - imputation / remove null values (null values will break the graph)
    - (TODO) normalize numerical features
    - dgl requires all types of nodes to start from 0, otherwise, the iterator will break
3. feature engineering at csv level
4. If data can’t be fixed internally, return an error message to help users fix the data externally.
