# Disclaimer:

# This code is a reference implementation extracted from production environment 
# and will not run directly. The competition dataset has certain truncations related 
# to personal data and internal ETL processes. Nevertheless, the code can be used 
# as a reference guide for similar implementations.


def proccessing(df_answer) -> pd.DataFrame:
    period_df = None
    for tariff_id, content, ranker_id in df_answer.values:
        json_data = json.loads(content)
            
        if json_data["routeData"]["searchRoute"].count("/") <= 1:
            v2 = 'segments' in json_data.keys()
            request_df = _json2table(json_data, v2)
            request_df["selected"] = (request_df["UID"] == tariff_id).apply(
                lambda x: 1 if x else 0
            )
            request_df["ranker_id"] = ranker_id
            if period_df is None:
                period_df = request_df
            else:
                period_df = pd.concat([period_df, request_df])
    return period_df


def __add_one(
    prefix: str, key: str, length: int, df: dict, cur_elem, count_pricings: int
) -> None:
    """
    Function for writing a specific field from json: None, int, float, str, list, etc.
    This function only receives lists with dict objects and dict variables themselves, as we break them down into
    smaller entities.
    params:
        prefix (str) - string indicating the parent in the json request structure
        key (str) - string name of the current field
        length (int) - since there are gaps in the data, and writing them one by one each time is slow -
        decided to fill them in chunks. Each time we get a new useful value, we fill all
        values before it with None. length stores the current position of the element being read,
        to fill all values before the current one with gaps (None).
        df (dict) - pointer to the table where the parsing result is stored
        cur_elem (any type) - value of the current element being added to the result table
        count_pricings (int) - number of proposed options with this record (record multiplier)
    """
    full_key = prefix + key

    if full_key in df:
        if len(df[full_key]) < length:
            df[full_key].extend([None] * (length - len(df[full_key])))
    else:
        df[full_key] = [None] * length

    df[full_key].extend([cur_elem] * count_pricings)


def __get_values(
    dict_inside: dict, df: dict, count_pricings: int, prefix: str, length: int
) -> None:
    """
    Function for writing a specific field from json: None, int, float, str, list, etc.
    This function only receives lists with dict objects and dict variables themselves, as we break them down into
    smaller entities.
    params:
        dict_inside (dict) - input dictionary to process
        df (dict) - pointer to the table where the parsing result is stored
        count_pricings (int) - number of proposed options with this record (record multiplier)
        prefix (str) - string indicating the parent in the json request structure
        length (int) - since there are gaps in the data, and writing them one by one each time is slow -
        decided to fill them in chunks. Each time we get a new useful value, we fill all
        values before it with None. length stores the current position of the element being read,
        to fill all values before the current one with gaps (None).
    """
    for key, value in dict_inside.items():
        # Skip all ids except UID
        if key == "id":
            # Save only one id for flight options
            if prefix + key in [
                "data_pricings_id",
                "data_$values_pricings_id",
                "segments_id",
            ]:
                if "UID" in df:
                    df["UID"].append(value)
                else:
                    df["UID"] = [value]

        if key == "$id":
            continue

        # Processing nested dictionaries
        if isinstance(value, dict):
            __get_values(value, df, count_pricings, prefix + key + "_", length)

        # Processing missing values
        elif value is None:
            __add_one(prefix, key, length, df, value, count_pricings)

        # inside legs there are several segments just like in values
        elif isinstance(value, list):
            if value:
                # Processing lists with dictionaries
                if isinstance(value[0], dict):
                    if key in ["pricings", "pricingInfo"]:
                        for item in value:
                            __get_values(item, df, 1, prefix + f"{key}_", length)
                    else:
                        for i, item in enumerate(value):
                            __get_values(
                                item, df, count_pricings, prefix + f"{key}{i}_", length
                            )

                # Processing lists with other values
                else:
                    __add_one(prefix, key, length, df, value, count_pricings)
            else:
                # Processing empty lists
                __add_one(prefix, key, length, df, value, count_pricings)
        # Processing remaining regular values
        else:
            __add_one(prefix, key, length, df, value, count_pricings)


def __get_rec_values(
    dict_inside: dict, df: dict, prefix: str, v2: bool = False
) -> None:
    """
    Main recursive function for parsing json file. Walks through nested dictionaries and lists and saves all data in df.
    params:
        dict_inside (dict) - Source dictionary. Contains information about client personal data,
        request data and proposed flight options. This is what we parse into table df.
        df (dict) - pointer to the table where the parsing result is stored
        prefix (str) - string indicating the parent in the json request structure
        v2 (bool) - JSON structure version
    """
    for key, value in dict_inside.items():
        if key in ["$id"]:
            continue

        # Processing personal data, etc.
        if isinstance(value, dict):
            __get_rec_values(value, df, prefix + key + "_", v2)

        elif value is None:
            df[prefix + key] = None

        # Processing flight options
        elif key == "$values":
            length = 0
            df["$values"] = {}
            for item in value:
                __get_values(
                    item,
                    df["$values"],
                    len(item["pricings"]),
                    prefix + key + "_",
                    length,
                )
                length += len(item["pricings"])

        # Processing flight options for second JSON version
        elif (key == "data") & v2:
            length = 0
            df["data"] = {}
            for item in value:
                __get_values(
                    item, df["data"], len(item["pricings"]), prefix + key + "_", length
                )
                length += len(item["pricings"])

        # Processing segment options
        elif (key == "segments") & v2:
            df["segments"] = {}
            for item in value:
                __get_values(item, df["segments"], 1, prefix + key + "_", 0)
        else:
            df[prefix + key] = value


def __rename_columns(df_columns: pd.DataFrame, v2: bool = False):
    """
    Function that renames columns to a more convenient form. Removes redundant parts from names,
    which are formed as a result of field inheritance within the json file.
        v2 (bool) - JSON structure version
    """
    # New version of naming
    prefixes_for_remove = ['routeData_', 'personalData_']
    
    if v2:
        prefixes_for_remove.append("data_pricings_")
        prefixes_for_remove.append("metadata_")
        prefixes_for_remove.append("data_")
    else:
        prefixes_for_remove.append("data_$values_pricings_")
        prefixes_for_remove.append("data_$values_")

    for each in prefixes_for_remove:
        df_columns = df_columns.str.replace(each, "")
    
    return df_columns


def combine_dataframe(d: dict) -> pd.DataFrame:
    """
    Converting dictionary to table with gap filling
    """
    # Number of proposed options
    length = len(d["UID"])

    # Fill missing data with None
    for key in d.keys():
        if len(d[key]) < length:
            d[key].extend([None] * (length - len(d[key])))

    # Convert dictionary to DataFrame
    return pd.DataFrame(d, columns=d.keys())


def _json2table(
    input_dict: dict, v2: bool = False, all_features: bool = False
) -> pd.DataFrame:
    """
    Converting dictionary to table with gap filling
    """
    # Dictionary for collecting data from source dictionary
    res_dict = {}
    # Dictionary for personal data
    personal_dict = {}

    # Start parsing json
    __get_rec_values(input_dict, res_dict, "", v2)

    # Write parsed personal data
    for key in res_dict.keys():
        if ((key != "$values") & (v2 == False)) | (
            (key != "data") & (key != "segments") & (v2)
        ):
            personal_dict[key] = res_dict.get(key)

    # Convert dictionary to DataFrame
    df = combine_dataframe(res_dict.get("data") if v2 else res_dict.get("$values"))

    fares_df = pd.DataFrame()
    fares_info_cols = [each for each in df.columns if "faresInfo" in each]
    intent = 4 if v2 else 5
    suffixes = set(["_".join(each.split("_")[intent:]) for each in fares_info_cols])

    # extract fare fields in breakdown by offers
    for i in range(8):
        if v2:
            fares_cols = [
                f"data_pricings_pricingInfo_faresInfo{i}_{suffix}"
                for suffix in suffixes
            ]
        else:
            fares_cols = [
                f"data_$values_pricings_pricingInfo_faresInfo{i}_{suffix}"
                for suffix in suffixes
            ]

        _tmp_df = df.reindex(columns=["UID"] + fares_cols)
        df = df.drop(columns=fares_cols, errors="ignore")
        _tmp_df.columns = ["UID"] + list(suffixes)
        _tmp_df = _tmp_df[_tmp_df["applyToSegmentIds"].notna()]
        fares_df = pd.concat([fares_df, _tmp_df], axis=0)
    fares_df = fares_df.explode("applyToSegmentIds")

    # Sometimes fare duplicates occur
    fares_df = fares_df.drop_duplicates(subset=["UID", "applyToSegmentIds"])

    # extract values about travel segments
    if v2:
        df_segments = combine_dataframe(res_dict.get("segments"))
        df_segments.columns = df_segments.columns.str.replace("segments_", "")

        segments_df = pd.DataFrame()
        for n_leg in range(MAX_LEGS):
            leg_id_name = f"data_legs{n_leg}_segments"
            if leg_id_name in df.columns:
                _tmp_df = df[["UID", leg_id_name]].explode(leg_id_name)
                _tmp_df = _tmp_df.merge(
                    df_segments,
                    left_on=leg_id_name,
                    right_on="UID",
                    how="left",
                    suffixes=["", "_seg"],
                )
                _tmp_df["n_leg"] = n_leg
                _tmp_df["n_segment"] = _tmp_df.groupby("UID").cumcount()
                _tmp_df = _tmp_df.drop(columns=[leg_id_name, "UID_seg"])
                df = df.drop(columns=[f"data_legs{n_leg}_segments"])
                segments_df = pd.concat([segments_df, _tmp_df], axis=0)
    else:
        suffixes = set(
            [
                "_".join(each.split("_")[4:])
                for each in df.columns
                if ("data_$values_legs" in each) & ("_segments" in each)
            ]
        )
        segments_df = pd.DataFrame()

        for n_leg, n_segment in product(range(MAX_LEGS), range(MAX_SEGMENTS)):
            seg_cols = [
                f"data_$values_legs{n_leg}_segments{n_segment}_{suffix}"
                for suffix in suffixes
            ]
            _tmp_df = df.reindex(columns=["UID"] + seg_cols)
            df = df.drop(columns=seg_cols, errors="ignore")
            _tmp_df.columns = ["UID"] + list(suffixes)
            _tmp_df = _tmp_df[_tmp_df["id"].notna()]
            _tmp_df["n_leg"] = n_leg
            _tmp_df["n_segment"] = n_segment
            segments_df = pd.concat([segments_df, _tmp_df], axis=0)

    # Merge segments and fares
    merged_df = segments_df.merge(
        fares_df,
        how="left",
        left_on=["UID", "id"],
        right_on=["UID", "applyToSegmentIds"],
        suffixes=["", "Fares"],
    )
    merged_df = merged_df.pivot(
        index=["UID", "n_leg"],
        columns=["n_segment"],
    ).reset_index()
    merged_df.columns = merged_df.columns.map(lambda x: f"segments{x[1]}_{x[0]}")
    merged_df = merged_df.pivot(
        index=["segments_UID"],
        columns=["segments_n_leg"],
    )
    merged_df.columns = merged_df.columns.map(lambda x: f"legs{x[1]}_{x[0]}")

    # Merge DataFrame with legs and travel segments
    df = df.merge(merged_df, how="left", left_on="UID", right_index=True)

    # Add personal data to DataFrame
    for column in personal_dict.keys():
        df[column] = personal_dict.get(column)
    df.columns = __rename_columns(df.columns, v2)
    
    if all_features != True:
        used_cols = (
            USED_COLS
            + [GROUP_COL, REQUEST_DATE]
            + [
                f"legs{legn}_{leg_col}"
                for legn, leg_col in product(range(MAX_LEGS), LEGS_COLS)
            ]
            + [
                f"legs{legn}_segments{segment_n}_{leg_col}"
                for legn, segment_n, leg_col in product(
                    range(MAX_LEGS), range(MAX_SEGMENTS), LEGS_SEGMENTS_COLS
                )
            ]
        )
        df = df.drop(columns=df.columns[df.columns.duplicated()])
        df = df.reindex(columns=used_cols)

    df = df.reindex(columns=sorted(df.columns))
    return df

