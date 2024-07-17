import pandas as pd
import numpy as np
from sklearn.tree import _tree
import pycaret.anomaly as py_caret_anom
import umap
import altair as alt
import altair as alt
alt.data_transformers.enable("vegafusion")
from tqdm.notebook import tqdm
import lime
import lime.lime_tabular


# https://mljar.com/blog/extract-rules-decision-tree/
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def re_bin_column(source_df, item_col="Field26", top_item_num=5, other_str="Other", drop_item_col=True,
                  drop_single_value_col=True):

    temp_df = source_df.copy()
    return_cols = list()

    temp_df[item_col] = [x.replace(" ", "_") for x in temp_df[item_col].values]

    top_item_counts = temp_df[item_col].value_counts()
    # print("top_item_counts :")
    # print(top_item_counts)
    len_top_item_counts = len(top_item_counts)
    if len_top_item_counts > top_item_num:

        top_items = top_item_counts[:(top_item_num - 1)].index
        # print("top_items :", top_items)
        re_labeled_col = [x if x in top_items else other_str for x in temp_df[item_col].values]

        print(f"    >> Inserting '{'Binned_' + item_col}' column...")
        temp_df["Binned_" + item_col] = re_labeled_col
        return_cols.append("Binned_" + item_col)

        if drop_item_col:
            print(f"    >> Removing column '{item_col}'...")
            temp_df = temp_df.drop(columns=[item_col])
        else:
            return_cols.append(item_col)

    elif len_top_item_counts == 1:
        if drop_single_value_col:
            print(f"    >> Removing column '{item_col}'...")
            temp_df = temp_df.drop(columns=[item_col])
    else:
        return_cols.append(item_col)

    return temp_df, return_cols


def prepare_data_columns(source_anomaly_df, top_item_num=5, other_str="Other",
                         ignore_cols=["UMAP_0", "UMAP_1", "Anomaly", "Anomaly_Score"]):

    print(f"  >> Original data has dimensions {source_anomaly_df.shape}")

    source_df = source_anomaly_df.copy()

    if len(ignore_cols) > 0:
        source_cols = source_df.columns
        alt_source_cols = [x for x in source_cols if x not in ignore_cols]
        ignore_data = source_df[ignore_cols]
        source_df = source_df[alt_source_cols]

    source_df = source_df.infer_objects()

    int_data = source_df.select_dtypes(include=["int"])
    int_cols = list(int_data.columns)
    return_int_cols = int_cols
    int_data_alt = pd.DataFrame()

    # print("len(int_cols) :", len(int_cols))
    # print("int_cols :", int_cols)
    # return_int_cols = list()
    # 
    # if len(int_data) > 0:
    #     for int_col in int_cols:
    #         try:
    #             source_df[int_col] = source_df[int_col].astype(float)
    #         except:

    #     int_data = int_data.astype("str")
    #     int_data_alt = int_data.copy()
    #     print(f"   >> Binning 'int' columns...")
    #     for int_col in int_cols:
    #         int_data_alt, temp_int_cols = re_bin_column(source_df=int_data_alt, item_col=int_col,
    #                                                     top_item_num=top_item_num, other_str=other_str,
    #                                                     drop_item_col=True, drop_single_value_col=True)
    #         return_int_cols.extend(temp_int_cols)

    str_data = source_df.select_dtypes(include=["O"])
    str_cols = list(str_data.columns)
    # print("str_cols :", str_cols)
    return_str_cols = list()
    if len(str_data) > 0:
        str_data_alt = str_data.copy()
        print(f"   >> Binning 'str' columns...")
        for str_col in str_cols:
            str_data_alt, temp_str_cols = re_bin_column(source_df=str_data_alt, item_col=str_col,
                                                        top_item_num=top_item_num, other_str=other_str,
                                                        drop_item_col=True, drop_single_value_col=True)
            return_str_cols.extend(temp_str_cols)

    time_data = source_df.select_dtypes(include=[np.datetime64])
    time_cols = list(time_data.columns)

    float_data = source_df.select_dtypes(include=["float"])
    float_cols = list(float_data.columns)

    # if len(time_date) > 0:
    #     return_df = time_date

    return_df = pd.concat([time_data, int_data_alt, str_data_alt, float_data, ignore_data], axis=1)
    # return_df = pd.concat([str_data_alt], axis=1)

    print(f"  >> New data has dimensions {return_df.shape}")

    return return_df, return_int_cols, return_str_cols, time_cols, float_cols


def plot_anomaly_chart(anomaly_df,
                       dim_0_col="UMAP_0",
                       dim_1_col="UMAP_1",
                       data_name="Records of X",
                       data_name_suffix="Anomalies",
                       size_col="Anomaly_Score",
                       color_column="Anomaly_Score",
                       color_scheme="yellowgreenblue",
                       opacity=0.35,
                       model_type="abod",
                       anomaly_chart_width=250,
                       anomaly_chart_height=250,
                       tool_tip_cols=["Anomaly_Score", "Anomaly"]):

    if size_col:
        size_alt = alt.Size(size_col, legend=None)
        print(f"  >> Generating anomaly chart...")
        anomaly_chart = alt.Chart(anomaly_df) \
            .mark_circle(filled=False, opacity=opacity) \
            .encode(x=dim_0_col + ":Q",
                    y=dim_1_col + ":Q",
                    size=size_alt,
                    color=alt.Color(color_column, scale=alt.Scale(scheme=color_scheme), legend=None),
                    tooltip=tool_tip_cols) \
            .properties(width=anomaly_chart_width, height=anomaly_chart_height,
                        title=data_name + " " + data_name_suffix + " ('" + model_type + "')") \
            .interactive()
    else:
        size_alt = None
        print(f"  >> Generating anomaly chart...")
        anomaly_chart = alt.Chart(anomaly_df) \
            .mark_circle(filled=False, opacity=opacity) \
            .encode(x=dim_0_col + ":Q",
                    y=dim_1_col + ":Q",
                    color=alt.Color(color_column, scale=alt.Scale(scheme=color_scheme), legend=None),
                    tooltip=tool_tip_cols) \
            .properties(width=anomaly_chart_width, height=anomaly_chart_height,
                        title=data_name + " " + data_name_suffix + " ('" + model_type + "')") \
            .interactive()

    return anomaly_chart


def test_anomaly_model(source_df,
                       pycaret_exp,
                       model_type="abod",
                       fraction=0.05,
                       size_col="Anomaly_Score",
                       color_column="Anomaly_Score",
                       data_name="Truck Haul Cycles",
                       data_name_suffix="Anomalies",
                       color_scheme="yellowgreenblue",
                       opacity=0.25,
                       tooltip=["Field17", "Field86", "Field56", "Field52", "Field90"],
                       anomaly_chart_width=250,
                       anomaly_chart_height=250,
                       random_state=39):

    print(f"  >> Creating anomaly model '{model_type}'...")

    py_caret_anom.set_current_experiment(pycaret_exp)
    anomaly_model = py_caret_anom.create_model(model_type, fraction=fraction)

    print(f"  >> Generating predictions...")
    anomaly_predictions = py_caret_anom.predict_model(model=anomaly_model, data=source_df)

    X_transformed = py_caret_anom.get_config("X_transformed")

    print(f"  >> Fitting UMAP...")
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(X_transformed)
    embedding_df = pd.DataFrame(embedding, columns=["UMAP_0", "UMAP_1"], index=X_transformed.index)

    anomaly_predictions_df = source_df.merge(embedding_df, how="left", left_index=True, right_index=True)
    anomaly_predictions_df = anomaly_predictions_df.merge(anomaly_predictions[["Anomaly", "Anomaly_Score"]],
                                                          how="left",
                                                          left_index=True, right_index=True)

    tooltip_alt = ["Anomaly_Score", "Anomaly"]
    tooltip_alt.extend(tooltip)

    anomaly_chart = plot_anomaly_chart(anomaly_df=anomaly_predictions_df,
                                       dim_0_col="UMAP_0",
                                       dim_1_col="UMAP_1",
                                       data_name=data_name,
                                       data_name_suffix=data_name_suffix,
                                       size_col=size_col,
                                       color_column=color_column,
                                       color_scheme=color_scheme,
                                       opacity=opacity,
                                       model_type=model_type,
                                       anomaly_chart_width=anomaly_chart_width,
                                       anomaly_chart_height=anomaly_chart_height,
                                       tool_tip_cols=tooltip_alt)

    return anomaly_predictions_df, anomaly_chart, anomaly_model


def get_col_unique_counts(source_df):

    unique_dict = {}
    col_list = list(source_df.columns)

    for col in col_list:
        temp_num_unique = len(source_df[col].unique())
        unique_dict[col] = temp_num_unique

    return unique_dict


def get_unique_col(source_df):

    df_len = len(source_df)

    unique_dict = get_col_unique_counts(source_df=source_df)

    possible_unique_key_list = []
    for col, count in unique_dict.items():
        if count == df_len:
            possible_unique_key_list.append(col)

    return possible_unique_key_list


def find_null_columns(source_df):

    df_len = len(source_df)

    df_nulls = source_df.isnull().sum()
    perc_null_series = (df_nulls / df_len)
    # perc_null_series.sort_values(ascending=False)
    perc_null_dict = perc_null_series.to_dict()

    return perc_null_dict


def remove_null_cols_by_threshold(source_df, perc_null_dict, null_threshold=0.01):

    null_cols = [key for key, value in perc_null_dict.items() if value > null_threshold]

    return_df = source_df.copy()

    for null_col in null_cols:
        try:
            return_df = return_df.drop(columns=[null_col])
        except:
            print(f"  >> Could not drop column: '{null_col}'")

    return return_df, null_cols


def test_many_anomaly_models(source_df,
                             pycaret_exp,
                             anomaly_model_list=["cluster", "iforest", "knn", "histogram"],
                             tooltip=["Field17", "Field86", "Field56", "Field52", "Field90"],
                             color_scheme="yellowgreenblue",
                             color_column="Anomaly",
                             data_name="Food Delivery Records",
                             data_name_suffix="Characteristics",
                             opacity=0.25,
                             anomaly_chart_cols=3,
                             anomaly_chart_width=250,
                             anomaly_chart_height=250,
                             anomaly_fractions=[0.05, 0.10],
                             random_state=2984):

    temp_anomaly_row_charts = []
    temp_anomaly_col_charts = []
    return_dict = {}

    for model_type in tqdm(anomaly_model_list):
        for anomaly_fraction in anomaly_fractions:
            # try:
            anomaly_predictions_df, anomaly_chart, anomaly_model = \
                test_anomaly_model(source_df=source_df,
                                   pycaret_exp=pycaret_exp,
                                   model_type=model_type,
                                   fraction=anomaly_fraction,
                                   size_col=None,
                                   color_column=color_column,
                                   data_name=data_name,
                                   data_name_suffix=data_name_suffix + f" ({anomaly_fraction:.0%} Outliers)",
                                   color_scheme=color_scheme,
                                   opacity=opacity,
                                   tooltip=tooltip,
                                   anomaly_chart_width=anomaly_chart_width,
                                   anomaly_chart_height=anomaly_chart_height,
                                   random_state=random_state)

            return_dict[model_type] = (anomaly_predictions_df, anomaly_chart, anomaly_model)

            temp_anomaly_row_charts.append(anomaly_chart)

            if len(temp_anomaly_row_charts) == anomaly_chart_cols:
                temp_row = alt.hconcat(*temp_anomaly_row_charts)
                temp_anomaly_col_charts.append(temp_row)
                temp_anomaly_row_charts = []

                # cluster_anomaly_chart.display()
            # except:
            #     print(f"  >> Could not create anomaly model '{model_type}'.")

    if len(temp_anomaly_row_charts) > 0:
        temp_row = alt.hconcat(*temp_anomaly_row_charts).resolve_scale(size='independent', color='independent')
        temp_anomaly_col_charts.append(temp_row)
        temp_anomaly_row_charts = []

    combined_chart = alt.vconcat(*temp_anomaly_col_charts)
    combined_chart = combined_chart.resolve_scale(size='independent', color='independent')

    return return_dict, combined_chart


def get_lime_explanations_for_anomalies(transformed_data_df, model_type,
                                        lime_num_features=7,
                                        anomaly_fraction=0.05,
                                        anomaly_kernel_width=3,
                                        random_state=349):

    print(f"  >> Creating anomaly model with transformed data...")
    lime_anomaly_experiment = py_caret_anom.setup(data=transformed_data_df,
                                                  session_id=random_state,
                                                  # max_encoding_ohe=11,
                                                  preprocess=False,
                                                  imputation_type="simple",
                                                  numeric_imputation="mean",
                                                  categorical_imputation="mode")

    lime_anomaly_model = py_caret_anom.create_model(model_type, fraction=anomaly_fraction)

    print(f"  >> Generating predictions...")
    lime_anomaly_predictions = py_caret_anom.predict_model(model=lime_anomaly_model, data=transformed_data_df)

    print(f"  >> Creating 'LimeTabularExplainer'...")
    lime_anomaly_explainer = lime.lime_tabular.LimeTabularExplainer(transformed_data_df.values,
                                                                    feature_names=list(transformed_data_df.columns),
                                                                    class_names=[0, 1],
                                                                    kernel_width=anomaly_kernel_width,
                                                                    mode="classification")

    lime_anomaly_predictions_alt = lime_anomaly_predictions[lime_anomaly_predictions["Anomaly"] == 1]
    anomaly_records = lime_anomaly_predictions_alt.sort_values(["Anomaly_Score"], ascending=False)
    anomaly_sccores = lime_anomaly_predictions_alt["Anomaly_Score"]
    anomaly_records = anomaly_records.drop(columns=["Anomaly", "Anomaly_Score"])

    print(f"  >> There are {len(anomaly_records)} outlier records...")
    # anomaly_records_unseen_data = transformed_data_df.loc[anomaly_records_index]

    # print("anomaly_records.shape :", anomaly_records.shape)

    return_dict = dict()
    single_record_exp_list = list()
    for anomaly_score, (idx, row) in tqdm(zip(anomaly_sccores, anomaly_records.iterrows()), total=len(anomaly_records)):

        single_record_exp = lime_anomaly_explainer.explain_instance(row.values,
                                                                    lime_anomaly_model.predict_proba,
                                                                    num_features=lime_num_features)
        single_record_exp_list.append(single_record_exp)

        as_list = single_record_exp.as_list()

        positive_influence_on_anomaly = list()
        negative_influence_on_anomaly = list()

        for feature, value in as_list:
            if value > 0.0:
                positive_influence_on_anomaly.append(feature)
            else:
                negative_influence_on_anomaly.append(feature)

        return_dict[idx] = {"as_list": as_list,
                            "predict_proba": single_record_exp.predict_proba,
                            "class_names": single_record_exp.class_names,
                            "anomaly_score": anomaly_score,
                            "positive_influence_on_anomaly": positive_influence_on_anomaly,
                            "negative_influence_on_anomaly": negative_influence_on_anomaly}

    return return_dict, single_record_exp_list, lime_anomaly_predictions, lime_anomaly_predictions_alt, lime_anomaly_model, \
           lime_anomaly_experiment, lime_anomaly_explainer




