# csv_preprocessing.py

import sys
import os
import pandas as pd
import numpy as np
import uuid


def check_columns(
    df,
    user_node,
    product_node,
    user_features,
    product_features,
    edge_features,
    new_features,
):
    """
    Check if all the specified columns exist in the data. Return dataframe that contains only the selected columns.
    :return: boolean
    """
    exist_columns = df.columns.tolist()

    selected_columns = []
    selected_columns.extend([user_node, product_node])
    selected_columns.extend(user_features)
    selected_columns.extend(product_features)
    selected_columns.extend(edge_features)

    required_features_for_new_features = []
    for k in new_features.keys():
        required_features_for_new_features.extend([v for v in new_features[k]])

    for feature in selected_columns:
        if not feature in exist_columns:
            raise AttributeError(
                'Specified feature `{}` is not in the uploaded data'.format(
                    feature
                )
            )

    for feature in required_features_for_new_features:
        if not feature in exist_columns:
            raise AttributeError(
                'Specified feature `{}` is not in the uploaded data'.format(
                    feature
                )
            )

    return df[set(selected_columns + required_features_for_new_features)]


def remove_duplicate(df):
    """
    Remove deplicate records. Report duplicate percentage when it's higher than 20%.
    :return: deduped dataframe
    """
    N = df.shape[0]
    df.drop_duplicates(keep='last', inplace=True)
    N_deduped = df.shape[0]
    dup_pct = (((N - N_deduped) / N) * 100) // 1

    if not dup_pct <= 20:
        print(
            'Warning: Duplicate records take up more than 20% of the uploaded data. Duplicate percentage: {}'.format(
                dup_pct
            )
        )

    return df


def handling_nulls(df):
    """
    Stategies:
    * user_node and product_node: remove all null records.
    * Numerical features: (1) drop the column if null_pct >=10%. (2) impute using median if 3%< null_pct <10%. (3) remove the record if null_pct <=3%.
    * Categorical features: (1) drop the column if null_pct >=5%. (2) remove the record if null_pct <5%.
    :return: dataframe
    """
    df.dropna(subset=[USER_NODE, PRODUCT_NODE], inplace=True)

    remove_columns = []
    impute_median_columns = []

    categorical_list = list(
        set(df.columns) - set(df.select_dtypes(include=np.number).columns)
    )
    categorical_null_pct = df[categorical_list].isna().sum() / df.shape[0]
    remove_columns.extend(
        categorical_null_pct[categorical_null_pct >= 0.05].index.tolist()
    )

    numerical_null_pct = (
        df.select_dtypes(include=np.number).isna().sum() / df.shape[0]
    )
    remove_columns.extend(
        numerical_null_pct[numerical_null_pct >= 0.1].index.tolist()
    )
    impute_median_columns.extend(
        numerical_null_pct[
            (numerical_null_pct < 0.1) & (numerical_null_pct >= 0.03)
        ].index.tolist()
    )

    for feature in impute_median_columns:  # impute columns
        df[feature].fillna(df[feature].median(), inplace=True)
    df.drop(columns=remove_columns, inplace=True)  # drop columns
    df.dropna(
        inplace=True
    )  # drop all the records that still contains null value

    return df


def feature_engineering(df, new_features):
    """
    Create the assigned features. Subject to change.
    :return: dataframe
    """

    if "on_time_delivery" in new_features.keys():
        delivery_date, order_date = new_features["on_time_delivery"]
        on_time = pd.to_datetime(df[delivery_date]) < pd.to_datetime(
            df[order_date]
        )
        df['on_time_delivery'] = on_time.astype('int')

    if "avg_product_price_in_order" in new_features.keys():
        order_id, price = new_features["avg_product_price_in_order"]
        order_df = pd.DataFrame(
            df.groupby(order_id)[price].mean()
        ).reset_index()
        order_df.columns = [order_id, 'avg_product_price_per_order']
        df = pd.merge(df, order_df, on=order_id)

    if "freight_ratio_in_order" in new_features.keys():
        order_id, price, freight = new_features["freight_ratio_in_order"]
        ratio_df = pd.DataFrame(df.groupby(order_id)[price].sum()).reset_index()
        ratio_df.columns = [order_id, 'total_value_of_order']
        ratio_df = pd.merge(ratio_df, df[[order_id, freight]], on=order_id)
        ratio_df["freight_ratio_in_order"] = (
            ratio_df[freight] / ratio_df["total_value_of_order"]
        )
        df = pd.merge(
            df, ratio_df[[order_id, "freight_ratio_in_order"]], on=order_id
        )

    return df


def assign_node_id(df, user_node, product_node):
    """
    Assign node id as a continuous integer series starting from 0.
    """
    # Create a dictionary that maps UUIDs to integer IDs
    uuid_to_int = {}
    for i, u in enumerate(df[user_node].unique()):
        uuid_to_int[u] = i
    # Map the UUIDs to integer IDs using the dictionary
    df['node_id_user'] = df[user_node].map(uuid_to_int)

    uuid_to_int = {}
    for i, u in enumerate(df[product_node].unique()):
        uuid_to_int[u] = i
    df['node_id_product'] = df[product_node].map(uuid_to_int)

    return df


# params
CSV_IN_PATH = dataset_file_path + 'ecommerce_data.csv'
USER_NODE = "customer_id"
PRODUCT_NODE = "product_id"
USER_FEATURES = ["customer_zip_code_prefix", "customer_city", "customer_state"]
PRODUCT_FEATURES = [
    "product_category_name_english",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]
EDGE_TYPE = "purchase_review"
EDGE_FEATURES = [
    "price",
    "freight_value",
    "order_status",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "payment_sequential",
    "payment_type",
    "payment_installments",
    "payment_value",
    "review_score",
    "review_comment_message",
]
NEW_FEATURES = {
    "on_time_delivery": [
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
    "freight_ratio_in_order": ["order_id", "price", "freight_value"],
    "avg_product_price_in_order": ["order_id", "price"],
}
CSV_OUT_PATH = dataset_file_path + 'ecommerce_data_out.csv'

# features that will be passed to the graph:
# user_node_id, product_node_id
# USER_FEATURES, PRODUCT_FEATURES, EDGE_FEATURES
# NEW_FEATURES.keys()


if __name__ == '__main__':
    df = pd.read_csv(CSV_IN_PATH)
    df = check_columns(
        df,
        USER_NODE,
        PRODUCT_NODE,
        USER_FEATURES,
        PRODUCT_FEATURES,
        EDGE_FEATURES,
        NEW_FEATURES,
    )
    df = remove_duplicate(df)
    df = handling_nulls(df)
    df = feature_engineering(df, NEW_FEATURES)
    df = assign_node_id(df, USER_NODE, PRODUCT_NODE)
    df.to_csv(CSV_OUT_PATH, index=False)
