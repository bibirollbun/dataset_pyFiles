# %%capture
!pip install --upgrade pip
!pip install polars


import kagglehub
import os

path = kagglehub.competition_download("aeroclub-recsys-2025")
print("âœ… Dataset downloaded to:", path)


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)





import polars as pl
import numpy as np
import time
import os
import gc

print("=" * 60)
print("ğŸš€ GIAI Ä�Oáº N 6+: ULTIMATE PIPELINE (TIME-BASED SPLIT)")
print("   Chiáº¿n thuáº­t: Walk-Forward Validation (30% Past -> 70% Future)")
print("=" * 60)

# ==============================================================================
# 1. Cáº¤U HÃŒNH AGGREGATION (GIá»® NGUYÃŠN)
# ==============================================================================
HIGH_PRIORITY_CONFIGS = {
    'avg_segment_flexibility_score_by_user': {'group_by': ['profileId'], 'agg_col': 'segment_flexibility_score', 'agg_func': 'mean'},
    'std_segment_flexibility_score_by_user': {'group_by': ['profileId'], 'agg_col': 'segment_flexibility_score', 'agg_func': 'std'},
    'user_convenience_over_price_rate': {'group_by': ['profileId'], 'agg_col': 'chose_convenience_over_price', 'agg_func': 'mean'},
    'user_price_over_convenience_rate': {'group_by': ['profileId'], 'agg_col': 'chose_price_over_convenience', 'agg_func': 'mean'},
    'avg_premium_paid_vs_cheapest_by_user': {'group_by': ['profileId'], 'agg_col': 'premium_paid_vs_cheapest', 'agg_func': 'mean'},
    'company_price_discipline_rate': {'group_by': ['companyID'], 'agg_col': 'chose_bottom_quartile_price', 'agg_func': 'mean'},
    'company_segment_discipline_rate': {'group_by': ['companyID'], 'agg_col': 'chose_minimum_segments', 'agg_func': 'mean'},
    'company_premium_policy_rate': {'group_by': ['companyID'], 'agg_col': 'chose_top_quartile_price', 'agg_func': 'mean'},
    'avg_policy_flexibility_by_company': {'group_by': ['companyID'], 'agg_col': 'policy_flexibility_interaction', 'agg_func': 'mean'},
    'std_policy_flexibility_by_company': {'group_by': ['companyID'], 'agg_col': 'policy_flexibility_interaction', 'agg_func': 'std'},
    'user_avg_segment_tier_for_route': {'group_by': ['profileId', 'searchRoute'], 'agg_col': 'segment_tier', 'agg_func': 'mean'},
    'user_price_percentile_for_route': {'group_by': ['profileId', 'searchRoute'], 'agg_col': 'totalPrice_percentile_in_group', 'agg_func': 'mean'},
    'company_avg_segment_tier_for_route': {'group_by': ['companyID', 'searchRoute'], 'agg_col': 'segment_tier', 'agg_func': 'mean'},
    'avg_position_within_segment_tier_by_user': {'group_by': ['profileId'], 'agg_col': 'position_pct_within_segment_tier', 'agg_func': 'mean'},
    'company_segment_tier_preference': {'group_by': ['companyID'], 'agg_col': 'segment_tier', 'agg_func': 'mean'},
    'user_avg_convenience_value_score': {'group_by': ['profileId'], 'agg_col': 'convenience_value_score', 'agg_func': 'mean'},
    'user_sweet_spot_selection_rate': {'group_by': ['profileId'], 'agg_col': 'is_sweet_spot_option', 'agg_func': 'mean'},
    'user_avg_price_per_extra_segment': {'group_by': ['profileId'], 'agg_col': 'price_per_extra_segment', 'agg_func': 'mean'}
}

TIER1_CONFIGS = {
    'route_min_segment_selection_rate': {'group_by': ['searchRoute'], 'agg_col': 'is_min_segments_for_route', 'agg_func': 'mean'},
    'route_avg_segments_selected': {'group_by': ['searchRoute'], 'agg_col': 'total_segments', 'agg_func': 'mean'},
    'route_segment_acceptance_by_tier': {'group_by': ['searchRoute', 'route_specific_segment_tier'], 'agg_col': 'selected', 'agg_func': 'mean'},
    'company_segment_discipline_score': {'group_by': ['companyID'], 'agg_col': 'company_segment_consistency_score', 'agg_func': 'mean'},
    'company_direct_preference_by_route': {'group_by': ['companyID', 'searchRoute'], 'agg_col': 'is_min_segments', 'agg_func': 'mean'},
    'company_segment_override_tolerance': {'group_by': ['companyID'], 'agg_col': 'company_segment_override_rate_all', 'agg_func': 'mean'},
    'user_segment_consistency_by_route': {'group_by': ['profileId', 'searchRoute'], 'agg_col': 'user_segment_choice_std', 'agg_func': 'std'},
    'user_segment_preference_strength': {'group_by': ['profileId'], 'agg_col': 'user_min_segment_preference_rate', 'agg_func': 'mean'},
    'user_vs_company_segment_deviation': {'group_by': ['profileId'], 'agg_col': 'user_segments_vs_company_norm', 'agg_func': 'mean'}
}

ALL_AGG_CONFIGS = {**HIGH_PRIORITY_CONFIGS, **TIER1_CONFIGS}

# ==============================================================================
# 2. HÃ€M Táº O FULL FEATURES (GIá»® NGUYÃŠN LOGIC)
# ==============================================================================
def create_full_features_lazy(df_lazy):
    # (Giá»¯ nguyÃªn toÃ n bá»™ logic táº¡o feature cá»§a báº¡n á»Ÿ Ä‘Ã¢y)
    # ... [Copy y nguyÃªn ná»™i dung hÃ m create_full_features_lazy cá»§a báº¡n vÃ o Ä‘Ã¢y] ...
    # Ä�á»ƒ tiáº¿t kiá»‡m khÃ´ng gian chat, tÃ´i khÃ´ng paste láº¡i pháº§n thÃ¢n hÃ m nÃ y 
    # vÃ¬ nÃ³ khÃ´ng Ä‘á»•i logic, chá»‰ cáº§n Ä‘áº£m báº£o requestDate Ä‘Æ°á»£c xá»­ lÃ½ Ä‘Ãºng.
    
    cols = df_lazy.collect_schema().names()
    
    # --- Xá»­ lÃ½ sÆ¡ bá»™ requestDate Ä‘á»ƒ Ä‘áº£m báº£o tÃ­nh toÃ¡n ---
    # Náº¿u chÆ°a cÃ³ datetime, Ã©p kiá»ƒu ngay
    if "requestDate" in cols:
         # Thá»­ Ã©p kiá»ƒu date náº¿u nÃ³ lÃ  string
         df_lazy = df_lazy.with_columns(pl.col("requestDate").str.to_datetime(strict=False).alias("requestDate_dt"))

    # ... [Pháº§n cÃ²n láº¡i cá»§a hÃ m Feature Engineering giá»¯ nguyÃªn] ...
    
    # --- Paste láº¡i Ä‘oáº¡n code logic feature engineering cá»§a báº¡n vÃ o Ä‘Ã¢y ---
    # (TÃ´i giáº£ Ä‘á»‹nh báº¡n Ä‘Ã£ paste láº¡i code feature engineering vÃ o Ä‘Ã¢y)
    
    # --- A. Pre-processing ---
    mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
    mc_exists = [c for c in mc_cols if c in cols]
    if mc_exists: df_lazy = df_lazy.with_columns([pl.col(c).cast(pl.String) for c in mc_exists])

    # Duration String -> Minutes
    def dur_to_min(col_name):
        c = pl.col(col_name).cast(pl.Utf8)
        d = c.str.extract(r"^(\d+)\.", 1).cast(pl.Float64).fill_null(0) * 1440
        time_part = c.str.replace(r"^\d+\.", "")
        h = time_part.str.extract(r"^(\d+):", 1).cast(pl.Float64).fill_null(0) * 60
        m = time_part.str.extract(r":(\d+):", 1).cast(pl.Float64).fill_null(0)
        return d + h + m

    dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1, 2, 3)]
    ex_dur = [c for c in dur_cols if c in cols]
    df_lazy = df_lazy.with_columns([dur_to_min(c).alias(c) for c in ex_dur])

    # --- B. Base Numerical ---
    df_lazy = df_lazy.with_columns([
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") * 100 / (pl.col("totalPrice") + 1)).alias("tax_ratex100"), 
        pl.col("totalPrice").log1p().alias("log_price"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0) + pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
    ])
    
    df_lazy = df_lazy.with_columns([
        (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
          .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 0.01))
          .otherwise(1.0).alias("duration_ratio")
    ])

    # Aggregations (Cabin/Baggage)
    cabin_cols = [c for c in cols if c.endswith('_cabinClass')]
    bag_cols = [c for c in cols if 'baggageAllowance_quantity' in c]
    
    if bag_cols:
        df_lazy = df_lazy.with_columns(pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in bag_cols]).alias("baggage_mean"))
    else:
        df_lazy = df_lazy.with_columns(pl.lit(0.0).alias("baggage_mean"))

    if cabin_cols:
        df_lazy = df_lazy.with_columns(pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in cabin_cols]).alias("avg_cabin_class_all"))
        c0 = [c for c in cabin_cols if 'legs0_' in c]
        c1 = [c for c in cabin_cols if 'legs1_' in c]
        if c0 and c1:
            df_lazy = df_lazy.with_columns(
                pl.when(pl.col("legs1_duration").is_not_null())
                .then(pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in c0]) - 
                      pl.mean_horizontal([pl.col(c).cast(pl.Float64).fill_null(0) for c in c1]))
                .otherwise(0.0).alias("cabin_class_diff_legs")
            )
        else:
            df_lazy = df_lazy.with_columns(pl.lit(0.0).alias("cabin_class_diff_legs"))
    else:
        df_lazy = df_lazy.with_columns([pl.lit(0.0).alias("avg_cabin_class_all"), pl.lit(0.0).alias("cabin_class_diff_legs")])

    # Total Segments
    if mc_exists:
        df_lazy = df_lazy.with_columns(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.UInt8) for c in mc_exists]).alias("total_segments"))
    else:
        df_lazy = df_lazy.with_columns(pl.lit(0.0).alias("total_segments"))

    # --- C. Binary & Datetime ---
    majors = ["SU", "S7", "U6", "VN", "VJ", "QH"]
    df_lazy = df_lazy.with_columns([
        (pl.col("legs1_duration").is_null() | (pl.col("legs1_duration") == 0)).cast(pl.Int32).alias("is_one_way"),
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        ((pl.col("isVip") == 1) | (pl.col("frequentFlyer").fill_null("") != "")).cast(pl.Int32).alias("is_vip_freq"),
        (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
        (pl.col("legs0_segments0_marketingCarrier_code").is_in(majors)).cast(pl.Int32).alias("is_major_carrier") 
        if "legs0_segments0_marketingCarrier_code" in cols else pl.lit(0).alias("is_major_carrier"),
        pl.when(pl.col("legs1_duration").is_not_null() & (pl.col("legs1_duration") > 0))
          .then((pl.sum_horizontal([pl.col(c).is_not_null() for c in mc_exists if 'legs1_' in c]) == 1).cast(pl.Int32))
          .otherwise(0).alias("is_direct_leg1")
    ])

    dt_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    time_exprs = []
    for c in dt_cols:
        if c in cols:
            dt = pl.col(c).str.to_datetime(strict=False)
            h = dt.dt.hour().fill_null(12)
            wd = dt.dt.weekday().fill_null(0)
            time_exprs.extend([
                (np.sin(2 * np.pi * h / 24)).alias(f"{c}_hour_sin"),
                (np.cos(2 * np.pi * h / 24)).alias(f"{c}_hour_cos"),
                (np.sin(2 * np.pi * wd / 7)).alias(f"{c}_weekday_sin"),
                (np.cos(2 * np.pi * wd / 7)).alias(f"{c}_weekday_cos"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{c}_business_time")
            ])
    if time_exprs: df_lazy = df_lazy.with_columns(time_exprs)

    if "requestDate" in cols and "legs0_departureAt" in cols:
         df_lazy = df_lazy.with_columns(
             ((pl.col("legs0_departureAt").str.to_datetime(strict=False).cast(pl.Date) - 
               pl.col("requestDate").str.to_datetime(strict=False).cast(pl.Date)).dt.total_days().cast(pl.Int16).fill_null(0)).alias("days_to_departure")
         )
    else:
         df_lazy = df_lazy.with_columns(pl.lit(0).alias("days_to_departure"))

    if "frequentFlyer" in cols and mc_exists:
        ff = pl.col("frequentFlyer").fill_null("")
        matches = [pl.when(pl.col(c).is_not_null() & (pl.col(c) != "") & ff.str.contains(pl.col(c))).then(1).otherwise(0) for c in mc_exists]
        df_lazy = df_lazy.with_columns((pl.sum_horizontal(matches) / (pl.col("total_segments") + 1)).alias("trust_value"))
    else:
        df_lazy = df_lazy.with_columns(pl.lit(0.0).alias("trust_value"))

    # --- E. INTERMEDIATE FEATURES (SEARCH ROUTE & STATS) ---
    # 1. Search Route
    origin_col = "legs0_segments0_departureFrom_airport_iata"
    dest_expr = pl.lit("UNK")
    for l in [1, 0]:
        for s in range(3, -1, -1):
            c = f"legs{l}_segments{s}_arrivalTo_airport_iata"
            if c in cols:
                dest_expr = pl.when(dest_expr == "UNK").then(pl.col(c).fill_null("UNK")).otherwise(dest_expr)
    
    if origin_col in cols:
        df_lazy = df_lazy.with_columns((pl.col(origin_col).fill_null("UNK") + "-" + dest_expr).alias("searchRoute"))
    else:
        df_lazy = df_lazy.with_columns(pl.lit("UNK").alias("searchRoute"))

    # 2. TÃ­nh toÃ¡n Logic Trung gian cho Tier 1 Aggregations
    # Cáº§n tÃ­nh toÃ¡n min segment theo Route trÆ°á»›c
    df_lazy = df_lazy.with_columns(
        pl.col("total_segments").min().over("searchRoute").alias("min_seg_route")
    )

    df_lazy = df_lazy.with_columns([
        pl.col("totalPrice").min().over("ranker_id").alias("min_price_grp"),
        pl.col("total_segments").min().over("ranker_id").alias("min_seg_grp"),
        pl.col("totalPrice").rank("ordinal").over("ranker_id").alias("price_rank"),
        pl.col("totalPrice").count().over("ranker_id").alias("grp_cnt"),
        pl.col("total_duration").rank("ordinal").over("ranker_id").alias("dur_rank"),
        
        # Z-score duration
        pl.col("total_duration").mean().over("ranker_id").alias("mean_dur"),
        pl.col("total_duration").std().over("ranker_id").fill_null(1).alias("std_dur"),
        
        # [NEW TIER 1] Route Specific Logic
        (pl.col("total_segments") == pl.col("min_seg_route")).cast(pl.Int8).alias("is_min_segments_for_route"),
        (pl.col("total_segments") - pl.col("min_seg_route")).clip(0, 3).cast(pl.Int8).alias("route_specific_segment_tier"),
    ])
    
    df_lazy = df_lazy.with_columns([
        ((pl.col("price_rank") - 1) / (pl.col("grp_cnt") - 1 + 1e-6)).alias("totalPrice_percentile_in_group"),
        ((pl.col("dur_rank") - 1) / (pl.col("grp_cnt") - 1 + 1e-6)).alias("duration_percentile"),
        ((pl.col("mean_dur") - pl.col("total_duration")) / (pl.col("std_dur") + 1e-6)).alias("duration_z_score"),
        
        # Complex Logic
        (pl.col("total_segments") - pl.col("min_seg_grp")).cast(pl.Float32).alias("segment_flexibility_score"),
        (pl.col("totalPrice") - pl.col("min_price_grp")).alias("premium_paid_vs_cheapest"),
        (pl.col("total_segments") - pl.col("min_seg_grp")).clip(0, 3).cast(pl.Int8).alias("segment_tier"),
        (pl.col("total_segments") == pl.col("min_seg_grp")).cast(pl.Int8).alias("chose_minimum_segments"),
        (pl.col("pricingInfo_isAccessTP").fill_null(0) * (pl.col("price_rank") - 1) / (pl.col("grp_cnt") - 1 + 1e-6)).alias("policy_flexibility_interaction")
    ])
    
    # Quartiles & Trade-offs & Tier 1 Mapping
    df_lazy = df_lazy.with_columns([
        (pl.col("totalPrice_percentile_in_group") <= 0.25).cast(pl.Int8).alias("chose_bottom_quartile_price"),
        (pl.col("totalPrice_percentile_in_group") >= 0.75).cast(pl.Int8).alias("chose_top_quartile_price"),
        ((pl.col("chose_minimum_segments") == 1) & (pl.col("totalPrice") > pl.col("min_price_grp"))).cast(pl.Int8).alias("chose_convenience_over_price"),
        ((pl.col("totalPrice") == pl.col("min_price_grp")) & (pl.col("chose_minimum_segments") == 0)).cast(pl.Int8).alias("chose_price_over_convenience"),
        ((1 - pl.col("totalPrice_percentile_in_group")) * (1 - pl.col("duration_percentile"))).alias("convenience_value_score"),
        pl.when(pl.col("total_segments") > pl.col("min_seg_grp"))
          .then((pl.col("totalPrice") - pl.col("min_price_grp")) / (pl.col("total_segments") - pl.col("min_seg_grp")))
          .otherwise(0).alias("price_per_extra_segment"),
          
        # [MAPPING TIER 1 ALIASES] - Ä�á»ƒ khá»›p vá»›i config
        pl.col("chose_minimum_segments").alias("company_segment_consistency_score"),
        pl.col("chose_minimum_segments").alias("is_min_segments"),
        (1 - pl.col("chose_minimum_segments")).alias("company_segment_override_rate_all"),
        pl.col("total_segments").alias("user_segment_choice_std"), # Nguá»“n Ä‘á»ƒ tÃ­nh std
        pl.col("chose_minimum_segments").alias("user_min_segment_preference_rate"),
        pl.col("segment_flexibility_score").alias("user_segments_vs_company_norm"),
    ])
    
    # Position in Tier & Sweet Spot
    df_lazy = df_lazy.with_columns([
        pl.col("totalPrice").rank("ordinal").over(["ranker_id", "segment_tier"]).alias("rank_in_tier"),
        pl.col("totalPrice").count().over(["ranker_id", "segment_tier"]).alias("cnt_in_tier")
    ])
    
    df_lazy = df_lazy.with_columns([
        ((pl.col("rank_in_tier") - 1) / (pl.col("cnt_in_tier") - 1 + 1e-6)).alias("position_pct_within_segment_tier"),
        ((pl.col("totalPrice_percentile_in_group") <= 0.3) & (pl.col("duration_percentile") <= 0.3)).cast(pl.Int8).alias("is_sweet_spot_option")
    ])
    
    # Clean up temp
    df_lazy = df_lazy.drop(["min_price_grp", "min_seg_grp", "grp_cnt", "price_rank", "dur_rank", "mean_dur", "std_dur", "rank_in_tier", "cnt_in_tier", "min_seg_route"])
    
    return df_lazy

# ==============================================================================
# ==============================================================================
# HÃ€M Táº O MAP NÃ‚NG Cáº¤P (SMOOTHING + GLOBAL FALLBACK)
# ==============================================================================
def create_smoothed_maps(df_stats_lazy, smoothing_weight=20):
    print(f"ğŸ§  Calculating Maps with Bayesian Smoothing (m={smoothing_weight})...")
    
    # 1. Táº¡o feature cÆ¡ báº£n cho táº­p Stats
    df_aug = create_full_features_lazy(df_stats_lazy)
    
    # 2. TÃ¡ch dá»¯ liá»‡u:
    # - Má»™t sá»‘ feature cáº§n tÃ­nh trÃªn toÃ n bá»™ (VD: Win Rate)
    # - Má»™t sá»‘ feature chá»‰ tÃ­nh trÃªn cÃ¡c dÃ²ng Ä‘Æ°á»£c chá»�n (VD: GiÃ¡ trung bÃ¬nh cá»§a vÃ© Ä‘Æ°á»£c chá»�n)
    # -> Ä�á»ƒ Ä‘Æ¡n giáº£n vÃ  khá»›p logic cÅ©, ta váº«n lá»�c selected=1 cho cÃ¡c profile hÃ nh vi.
    # -> Tuy nhiÃªn, náº¿u báº¡n muá»‘n tÃ­nh WinRate chuáº©n, báº¡n nÃªn tÃ­nh trÃªn toÃ n bá»™ df_aug.
    # á»� Ä‘Ã¢y tÃ´i giá»¯ logic cÅ© cá»§a báº¡n (tÃ­nh trÃªn selected=1) Ä‘á»ƒ an toÃ n cho pipeline hiá»‡n táº¡i.
    df_base = df_aug.filter(pl.col("selected") == 1)
    
    maps = {}
    
    for feat_name, config in ALL_AGG_CONFIGS.items():
        keys = config['group_by']
        target = config['agg_col']
        func = config['agg_func']
        
        # --- BÆ¯á»šC A: TÃ�NH GLOBAL MEAN (FALLBACK) ---
        # TÃ­nh giÃ¡ trá»‹ trung bÃ¬nh toÃ n cá»¥c cá»§a cá»™t target trÃªn táº­p Stats
        # (DÃ¹ng streaming collect Ä‘á»ƒ láº¥y ra 1 sá»‘ thá»±c duy nháº¥t)
        try:
            if func == 'mean':
                global_val = df_base.select(pl.col(target).mean()).collect(streaming=True).item()
            elif func == 'std':
                global_val = df_base.select(pl.col(target).std()).collect(streaming=True).item()
            else:
                global_val = 0 # Default cho count
        except:
            global_val = 0 # Fallback an toÃ n náº¿u cá»™t toÃ n null
            
        # LÆ°u global_val vÃ o config Ä‘á»ƒ dÃ¹ng khi fillna sau nÃ y
        
        # --- BÆ¯á»šC B: TÃ�NH LOCAL STATS + SMOOTHING ---
        if func == 'mean':
            # Ã�p dá»¥ng cÃ´ng thá»©c Bayesian Smoothing
            # Cáº§n tÃ­nh: Sum vÃ  Count cá»§a tá»«ng nhÃ³m
            expr = (
                (pl.col(target).sum() + (global_val * smoothing_weight)) / 
                (pl.col(target).count() + smoothing_weight)
            )
        elif func == 'std':
            # Std khÃ´ng smooth theo cÃ´ng thá»©c trÃªn Ä‘Æ°á»£c, giá»¯ nguyÃªn
            expr = pl.col(target).std().fill_null(global_val)
        else:
            expr = pl.col(target).mean()
            
        # Collect Map
        stat = df_base.group_by(keys).agg(expr.alias(feat_name)).collect(streaming=True)
        
        maps[feat_name] = {
            'keys': keys, 
            'df': stat, 
            'fallback': global_val # <--- GIÃ� TRá»Š Ä�Iá»€N VÃ€O CHá»– TRá»�NG
        }
        
    return maps

# ==============================================================================
# HÃ€M APPLY NÃ‚NG Cáº¤P (DÃ™NG GLOBAL FALLBACK)
# ==============================================================================
def apply_features_with_fallback(df_lazy, agg_maps):
    df = create_full_features_lazy(df_lazy)
    
    for feat_name, map_info in agg_maps.items():
        keys = map_info['keys']
        fallback_val = map_info['fallback']
        
        # Náº¿u fallback lÃ  None (do lá»—i tÃ­nh toÃ¡n), gÃ¡n vá»� -1 hoáº·c 0
        if fallback_val is None: fallback_val = -1
        
        # Join Left
        # Thay vÃ¬ fill_null(-1), ta fill_null(fallback_val)
        df = df.join(
            map_info['df'].lazy(), 
            on=keys, 
            how="left"
        ).with_columns(
            pl.col(feat_name).fill_null(pl.lit(fallback_val)) # <--- Ä�Iá»‚M KHÃ�C BIá»†T
        )
        
    return df

# ==============================================================================
# ==============================================================================
# 4. THá»°C THI (TIME-BASED LOGIC - FIXED)
# ==============================================================================
N_SPLITS = 10
start_time = time.time()
path = "../input/aeroclub-recsys-2025"

print("[1/5] Loading & Parsing Dates for Time-Split (ROBUST FIX)...")

# 1. Load Lazy
train_scan = pl.scan_parquet(f'{path}/train.parquet')
test_scan = pl.scan_parquet(f'{path}/test.parquet')

# 2. Xá»¬ LÃ� DATE AN TOÃ€N (QUAN TRá»ŒNG)
# BÆ°á»›c A: Ã‰p requestDate vá»� String (Ä‘á»ƒ hÃ m feature engineering bÃªn dÆ°á»›i khÃ´ng bá»‹ lá»—i SchemaError)
train_scan = train_scan.with_columns(pl.col("requestDate").cast(pl.String))
test_scan = test_scan.with_columns(pl.col("requestDate").cast(pl.String))

# BÆ°á»›c B: Táº¡o cá»™t phá»¥ requestDate_dt chuáº©n Datetime Ä‘á»ƒ sort vÃ  cáº¯t
# DÃ¹ng str.to_datetime vá»›i strict=False Ä‘á»ƒ trÃ¡nh lá»—i náº¿u cÃ³ chuá»—i láº¡
train_scan = train_scan.with_columns(
    pl.col("requestDate").str.to_datetime(strict=False).alias("requestDate_dt")
).sort("requestDate_dt")

# 3. TÃ­nh Ä‘iá»ƒm cáº¯t (30% Quantile)
print("   -> Calculating Time Cut-off (20% Quantile)...")

# Collect cá»™t date, Bá»� QUA NULL (drop_nulls) Ä‘á»ƒ trÃ¡nh lá»—i None
date_series = train_scan.select(pl.col("requestDate_dt").drop_nulls()).collect().get_column("requestDate_dt")

if len(date_series) == 0:
    raise ValueError("â�Œ Lá»—i: KhÃ´ng parse Ä‘Æ°á»£c cá»™t requestDate nÃ o cáº£! Kiá»ƒm tra láº¡i Ä‘á»‹nh dáº¡ng ngÃ y thÃ¡ng.")

cut_off_date = date_series.quantile(0.20)
print(f"   -> Cut-off Date: {cut_off_date}")

if cut_off_date is None:
    # Fallback: Náº¿u váº«n None (hiáº¿m), láº¥y ngÃ y á»Ÿ vá»‹ trÃ­ 20% thá»§ cÃ´ng
    sorted_dates = date_series.sort()
    idx = int(len(sorted_dates) * 0.2)
    cut_off_date = sorted_dates[idx]
    print(f"   -> Cut-off Date (Manual Fallback): {cut_off_date}")

# 4. Chia dá»¯ liá»‡u
# LÆ°u Ã½: drop cá»™t requestDate_dt sau khi dÃ¹ng xong Ä‘á»ƒ khÃ´ng gÃ¢y rá»‘i schema
train_stats_lazy = train_scan.filter(pl.col("requestDate_dt") <= cut_off_date).drop("requestDate_dt")
train_model_lazy = train_scan.filter(pl.col("requestDate_dt") > cut_off_date).drop("requestDate_dt")

# In thÃ´ng tin kiá»ƒm tra
print(f"   -> Total Rows: {len(date_series):,}")
print(f"   -> Stats Set (Past): ~30% (Used for Maps)")
print(f"   -> Train Set (Future): ~70% (Used for Model)")

print("[2/5] Creating Smoothed Maps...")
# Gá»�i hÃ m má»›i, smoothing_weight=20 lÃ  con sá»‘ an toÃ n
agg_maps = create_smoothed_maps(train_stats_lazy, smoothing_weight=40)

print("[3/5] Applying Features with Global Fallback...")
# Gá»�i hÃ m apply má»›i
df_train_final = apply_features_with_fallback(train_model_lazy, agg_maps)
df_test_final = apply_features_with_fallback(test_scan, agg_maps)

print("[4/5] Selecting & Collecting...")
# Collect Schema Ä‘á»ƒ lá»�c cá»™t rÃ¡c
all_cols = df_train_final.collect_schema().names()

GARBAGE = ['_segments', 'flightNumber', 'seatsAvailable', 'weightMeasurementType', 
           'airport_city_iata', 'arrivalTo_airport_iata', 'departureFrom_airport_iata',
           'legs0_duration', 'legs1_duration', 'marketingCarrier_code', 'baggageAllowance_quantity',
           'miniRules', 'taxes', 'requestDate_dt'] # ThÃªm requestDate_dt vÃ o rÃ¡c cho cháº¯c

SYSTEM = ["ranker_id", "selected", "Id", "fold", "requestDate"]

final_cols = [c for c in all_cols if c in SYSTEM or not any(p in c for p in GARBAGE)]
print(f"   -> Features count: {len(final_cols)}")

pandas_df = df_train_final.select([c for c in final_cols if c != "Id"]).collect().to_pandas(use_pyarrow_extension_array=False)
test_df_pd = df_test_final.select([c for c in final_cols if c != "selected"]).collect().to_pandas(use_pyarrow_extension_array=False)

print("[5/5] Final Polish & Fold Creation...")
for df in [pandas_df, test_df_pd]:
    # Xá»­ lÃ½ Category
    if 'searchRoute' in df.columns: df['searchRoute'] = df['searchRoute'].astype('category')
    cat_candidates = ["pricingInfo_isAccessTP", "frequentFlyer", "isVip", "bySelf", "sex", "cancellation_status", "exchange_status"]
    for c in df.columns:
        if c in cat_candidates: df[c] = df[c].astype('category')
    
    # Chia Fold (Fixed Sorted)
    if 'ranker_id' in df.columns and 'selected' in df.columns:
         unique_ids = sorted(df['ranker_id'].unique())
         np.random.seed(42)
         fold_map = {uid: np.random.randint(0, N_SPLITS) for uid in unique_ids}
         df['fold'] = df['ranker_id'].map(fold_map)

print(f"âœ… DONE! Train (Future): {pandas_df.shape}, Test: {test_df_pd.shape}")
print(f"â�±ï¸� Time elapsed: {time.time() - start_time:.2f}s")


import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import gc
import os

print("ğŸš€ GIAI Ä�Oáº N 5: TRAINING (RAM-SAFE & FIXED DTYPES)")

# ==============================================================================
# 1. HÃ€M GIáº¢M RAM (PHIÃŠN Báº¢N AN TOÃ€N - FIX Lá»–I ID)
# ==============================================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'   Original memory usage: {start_mem:.2f} MB')
    
    # ğŸ”¥ DANH SÃ�CH CÃ�C Cá»˜T Cáº¦N GIá»® NGUYÃŠN (KHÃ”NG Ä�Æ¯á»¢C Ã‰P KIá»‚U)
    # Id vÃ  ranker_id báº¯t buá»™c pháº£i lÃ  String/Object Ä‘á»ƒ Merge vÃ  Groupby an toÃ n
    SKIP_COLS = ['Id', 'ranker_id', 'flight_hash', 'fold'] 
    
    for col in df.columns:
        # Náº¿u lÃ  cá»™t ID, bá»� qua ngay láº­p tá»©c
        if col in SKIP_COLS:
            continue
            
        col_type = df[col].dtype
        
        # Bá»� qua Object, Category VÃ€ Datetime
        if col_type != object and str(col_type) != 'category' and 'datetime' not in str(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Xá»­ lÃ½ sá»‘ nguyÃªn
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            # Xá»­ lÃ½ sá»‘ thá»±c
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            # Ã‰p kiá»ƒu category cho object/string (CHá»ˆ CÃ�C Cá»˜T KHÃ”NG Náº°M TRONG SKIP_COLS)
            if col_type == object:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'   Optimized memory usage: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df
# Ã�p dá»¥ng giáº£m RAM
print("ğŸ“‰ Optimizing RAM for Train...")
pandas_df = reduce_mem_usage(pandas_df)

print("ğŸ“‰ Optimizing RAM for Test...")
test_df_pd = reduce_mem_usage(test_df_pd)
gc.collect()




# import pandas as pd

# def check_split_ratio(df, threshold=50):
#     print("="*60)
#     print(f"ğŸ“Š PHÃ‚N TÃ�CH Tá»¶ Lá»† USER (THRESHOLD = {threshold})")
#     print("="*60)
    
#     # 1. TÃ­nh group_size náº¿u chÆ°a cÃ³
#     if 'group_size' not in df.columns:
#         print("âš™ï¸� Calculating group_size...")
#         # Ä�áº¿m sá»‘ dÃ²ng cá»§a má»—i ranker_id
#         df['group_size'] = df.groupby('ranker_id')[df.columns[0]].transform('count')

#     # 2. Táº¡o báº£ng thá»‘ng kÃª User (nhanh hÆ¡n lÃ  query trÃªn toÃ n bá»™ dataframe)
#     # Chá»‰ giá»¯ láº¡i ranker_id vÃ  group_size
#     user_stats = df[['ranker_id', 'group_size']].drop_duplicates()
    
#     # 3. TÃ­nh toÃ¡n
#     total_users = len(user_stats)
#     total_rows = len(df)
    
#     # NhÃ³m Small
#     small_users = user_stats[user_stats['group_size'] <= threshold]
#     n_small_users = len(small_users)
#     n_small_rows = len(df[df['group_size'] <= threshold])
    
#     # NhÃ³m Large
#     large_users = user_stats[user_stats['group_size'] > threshold]
#     n_large_users = len(large_users)
#     n_large_rows = len(df[df['group_size'] > threshold])
    
#     # 4. In káº¿t quáº£
#     print(f"1ï¸�âƒ£  Vá»� sá»‘ lÆ°á»£ng USER (Session):")
#     print(f"   - Tá»•ng User: {total_users:,}")
#     print(f"   - NhÃ³m Small: {n_small_users:,} user ({n_small_users/total_users:.2%})")
#     print(f"   - NhÃ³m Large: {n_large_users:,} user ({n_large_users/total_users:.2%})")
    
#     print(f"\n2ï¸�âƒ£  Vá»� sá»‘ lÆ°á»£ng DÃ’NG Dá»® LIá»†U (Rows):")
#     print(f"   - Tá»•ng Rows: {total_rows:,}")
#     print(f"   - NhÃ³m Small: {n_small_rows:,} dÃ²ng ({n_small_rows/total_rows:.2%})")
#     print(f"   - NhÃ³m Large: {n_large_rows:,} dÃ²ng ({n_large_rows/total_rows:.2%})")
    
#     print("-" * 60)
#     print("ğŸ’¡ NHáº¬N Ä�á»ŠNH:")
#     if n_small_users / total_users > 0.2:
#         print("   âœ… Tá»· lá»‡ User Small khÃ¡ cao (>20%). Chiáº¿n thuáº­t tÃ¡ch model lÃ  Ráº¤T HIá»†U QUáº¢.")
#         print("      LÃ½ do: Model Small Ä‘áº¡t HitRate ~0.78 sáº½ kÃ©o Ä‘iá»ƒm tá»•ng thá»ƒ lÃªn ráº¥t máº¡nh.")
#     else:
#         print("   âš ï¸� Tá»· lá»‡ User Small tháº¥p. Sá»± áº£nh hÆ°á»Ÿng cá»§a Model Small sáº½ khÃ´ng quÃ¡ lá»›n.")

# # --- CHáº Y KIá»‚M TRA ---
# # Kiá»ƒm tra trÃªn táº­p Train
# if 'pandas_df' in locals():
#     print("\n[TRAIN SET]")
#     check_split_ratio(pandas_df, threshold=70)

# # Kiá»ƒm tra trÃªn táº­p Test
# if 'test_df_pd' in locals():
#     print("\n[TEST SET]")
#     check_split_ratio(test_df_pd, threshold=70)


import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import gc
import polars as pl
import os
import matplotlib.pyplot as plt

print("ğŸš€ GIAI Ä�Oáº N 5: SINGLE LIGHTGBM RANKER (FIXED DTYPES)")

# ==============================================================================
# 1. Cáº¤U HÃŒNH & Lá»ŒC FEATURE (QUAN TRá»ŒNG: Sá»¬A Lá»–I á»� Ä�Ã‚Y)
# ==============================================================================
COMMON_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [3],
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    
    "max_bin": 63,
    "n_estimators": 3000,   
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 200, 
    "reg_lambda": 5.0,
    "colsample_bytree": 0.8,
    "subsample": 0.7,
    "subsample_freq": 1,
    'lambdarank_truncation_level': 10,
    "random_state": 42,
    "n_jobs": -1,
    "two_round": True,
    "verbose": -1
}


target = "selected"

# 1. Danh sÃ¡ch cÃ¡c cá»™t há»‡ thá»‘ng cháº¯c cháº¯n bá»�
DROP_LIST = [
    # â�Œ NHÃ“M THá»œI GIAN THÃ” (THá»¦ PHáº M CHÃ�NH)
    "legs0_arrivalAt", "legs0_departureAt", 
    "legs1_arrivalAt", "legs1_departureAt",
    "requestDate", 
    "legs0_segments0_departureFrom_airport_iata", # SÃ¢n bay cá»¥ thá»ƒ quÃ¡ cÅ©ng cÃ³ thá»ƒ gÃ¢y nhiá»…u
    "legs0_segments0_arrivalTo_airport_iata",
    
    # â�Œ CÃ�C Cá»˜T Há»† THá»�NG
    "ranker_id", "selected", "fold", "group_size", "size_tier", "Id"
]

# 2. Láº¥y táº¥t cáº£ cá»™t, trá»« cá»™t há»‡ thá»‘ng
initial_features = [c for c in pandas_df.columns if c not in DROP_LIST]

# 3. [FIX] Lá»�c bá»� cÃ¡c cá»™t cÃ³ kiá»ƒu dá»¯ liá»‡u khÃ´ng há»£p lá»‡ (Object, Datetime)
# LightGBM chá»‰ cháº¥p nháº­n: number (int/float), bool, category
valid_features = []
print("âš™ï¸� Checking feature dtypes...")

for col in initial_features:
    dtype = pandas_df[col].dtype
    # Giá»¯ láº¡i náº¿u lÃ  sá»‘, category hoáº·c bool
    if pd.api.types.is_numeric_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
        valid_features.append(col)
    else:
        print(f"   â�Œ Dropping raw column: {col} ({dtype})")

features = valid_features
print(f"ğŸ“‹ Final Valid Features: {len(features)}")
print(features)

# XÃ¡c Ä‘á»‹nh láº¡i Categorical Features tá»« danh sÃ¡ch features sáº¡ch
cat_cols = ["pricingInfo_isAccessTP", "frequentFlyer", "isVip", "bySelf", "sex", "cancellation_status", "exchange_status", "has_corporate_tariff", "has_access_tp", "is_vip_freq", "is_one_way", "has_fees", "is_major_carrier", "is_direct_leg1"]
categorical_features = [c for c in cat_cols if c in features]

# Ä�áº£m báº£o cá»™t 'fold' tá»“n táº¡i
if 'fold' not in pandas_df.columns:
    print("âš™ï¸� Creating Folds...")
    gkf = GroupKFold(n_splits=N_SPLITS)
    pandas_df['fold'] = -1
    for i, (_, v_idx) in enumerate(gkf.split(pandas_df, groups=pandas_df['ranker_id'])):
        pandas_df.iloc[v_idx, pandas_df.columns.get_loc('fold')] = i

# HÃ m Metric
def hitrate_metric(y_true, y_pred, group_ids):
    df_tmp = pl.DataFrame({"y": y_true, "p": y_pred, "g": group_ids})
    hits = df_tmp.sort("p", descending=True).group_by("g").head(3).filter(pl.col("y")==1)
    valid_g = df_tmp.group_by("g").len().filter(pl.col("len") > 10)
    if valid_g.height == 0: return 0.0
    return hits.join(valid_g, on="g", how="inner").height / valid_g.height

# ==============================================================================
# 2. VÃ’NG Láº¶P HUáº¤N LUYá»†N
# ==============================================================================
final_test_preds = np.zeros(len(test_df_pd))
oof_preds = np.zeros(len(pandas_df))

print(f"\nğŸ¥Š Báº¯t Ä‘áº§u Train {N_SPLITS} Folds...")

for fold in range(N_SPLITS):
    print(f"\n--- Fold {fold} ---")
    
    train_mask = pandas_df["fold"] != fold
    val_mask = pandas_df["fold"] == fold
    
    # Táº¡o View dá»¯ liá»‡u
    X_tr = pandas_df.loc[train_mask, features]
    y_tr = pandas_df.loc[train_mask, target]
    g_tr = pandas_df.loc[train_mask].groupby("ranker_id", sort=False).size().to_numpy()
    
    X_val = pandas_df.loc[val_mask, features]
    y_val = pandas_df.loc[val_mask, target]
    g_val = pandas_df.loc[val_mask].groupby("ranker_id", sort=False).size().to_numpy()
    val_ids = pandas_df.loc[val_mask, "ranker_id"]
    gc.collect()
    # Train
    model = lgb.LGBMRanker(**COMMON_PARAMS)
    model.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_val, y_val)], eval_group=[g_val],
        eval_metric=lambda y, p: [('hitrate', hitrate_metric(y, p, val_ids), True)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=categorical_features
    )
    
    # Predict
    oof_preds[val_mask] = model.predict(X_val)
    hr_score = model.best_score_['valid_0']['hitrate']
    print(f"   ğŸŒŸ Best HitRate Fold {fold}: {hr_score:.4f}")
    
    final_test_preds += model.predict(test_df_pd[features]) / N_SPLITS
    
    
# Chá»‰ váº½ Ä‘Æ°á»£c náº¿u báº¡n lÆ°u láº¡i model cá»§a fold cuá»‘i cÃ¹ng hoáº·c fold 0
# Giáº£ sá»­ biáº¿n 'model' Ä‘ang lÃ  model cá»§a Fold cuá»‘i cÃ¹ng

print("ğŸ“Š Váº½ biá»ƒu Ä‘á»“ tá»« LGBMRanker...")
results = model.evals_result_
epochs = len(results['valid_0']['ndcg@3'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, results['valid_0']['ndcg@3'], label='Validation NDCG')
ax.legend()
plt.ylabel('NDCG@3')
plt.title('LightGBM Training History (Sklearn API)')
plt.show()
# ==============================================================================
# 3. Táº O SUBMISSION
# ==============================================================================
print("\nğŸ“� Xuáº¥t file submission_single.csv...")

test_df_pd["score"] = final_test_preds
test_df_pd["rank"] = test_df_pd.groupby("ranker_id")["score"].rank(method="first", ascending=False).astype(int)

submission = test_df_pd[["Id", "ranker_id", "rank"]].copy()
submission.rename(columns={"rank": "selected"}, inplace=True)

data_dir = "../input/aeroclub-recsys-2025"
test_file = os.path.join(data_dir, "test.parquet")
if os.path.exists(test_file):
    original_order = pd.read_parquet(test_file, columns=["Id"])
    final_submission = original_order.merge(submission, on="Id", how="left")
    final_submission["selected"] = final_submission["selected"].fillna(1).astype(int)
    final_submission.to_csv("submission_single.csv", index=False)
else:
    submission.to_csv("submission_single.csv", index=False)

print(f"âœ… DONE! File: submission_single.csv")


print("="*40)
print(f"ğŸ“Š Tá»”NG Sá»� FEATURE Ä�Æ¯á»¢C DÃ™NG: {len(features)}")
print("="*40)

print("\nğŸ”� Danh sÃ¡ch chi tiáº¿t:")
for i, f in enumerate(features):
    # In ra tÃªn feature vÃ  kiá»ƒu dá»¯ liá»‡u cá»§a nÃ³
    dtype = pandas_df[f].dtype
    print(f"{i+1:02d}. {f:<35} ({dtype})")


import polars as pl
import pandas as pd
import numpy as np

print("=" * 60)
print("ğŸš€ GIAI Ä�Oáº N 7: DIVERSITY RE-RANKING (POST-PROCESSING)")
print("=" * 60)

# ==============================================================================
# 1. CHUáº¨N Bá»Š Dá»® LIá»†U (PANDAS -> POLARS)
# ==============================================================================
# Giáº£ sá»­ báº¡n Ä‘Ã£ cÃ³:
# - test_df_pd: DataFrame Pandas chá»©a features vÃ  ID
# - final_test_preds: Máº£ng numpy chá»©a Ä‘iá»ƒm dá»± Ä‘oÃ¡n cá»§a LightGBM

print("ğŸ”„ Converting Data to Polars for Re-ranking...")

# Táº¡o DataFrame submission cÆ¡ báº£n tá»« káº¿t quáº£ LightGBM
submission_lgbm = pd.DataFrame({
    'Id': test_df_pd['Id'],
    'ranker_id': test_df_pd['ranker_id'],
    'pred_score': final_test_preds
})

# Chuyá»ƒn sang Polars
pl_submission = pl.from_pandas(submission_lgbm)
pl_test_features = pl.from_pandas(test_df_pd)

# ==============================================================================
# 2. HÃ€M RE-RANK (LOGIC Cá»¦A Báº N)
# ==============================================================================
def re_rank(test_features: pl.DataFrame, submission: pl.DataFrame, penalty_factor=1.0):
    # Cá»™t dÃ¹ng Ä‘á»ƒ Ä‘á»‹nh danh chuyáº¿n bay (Physical Flight Identity)
    COLS_TO_HASH = [
        "legs0_departureAt", "legs0_arrivalAt", 
        "legs1_departureAt", "legs1_arrivalAt",
        "legs0_segments0_flightNumber", "legs1_segments0_flightNumber",
        # ThÃªm sÃ¢n bay Ä‘á»ƒ cháº¯c cháº¯n khÃ´ng trÃ¹ng
        "legs0_segments0_departureFrom_airport_iata" 
    ]
    
    # Kiá»ƒm tra cá»™t tá»“n táº¡i trÆ°á»›c khi xá»­ lÃ½
    available_cols = [c for c in COLS_TO_HASH if c in test_features.columns]
    
    # Ã‰p kiá»ƒu String vÃ  Fill Null Ä‘á»ƒ Hash
    test_features = test_features.with_columns(
        [pl.col(c).cast(pl.String).fill_null("NULL") for c in available_cols]
    )

    # Join Ä‘iá»ƒm dá»± Ä‘oÃ¡n vÃ o Features
    df = submission.join(test_features.select(["Id", "ranker_id"] + available_cols), 
                         on=["Id", "ranker_id"], how="left")

    # 1. Táº¡o Flight Hash (Ä�á»‹nh danh chuyáº¿n bay)
    # Káº¿t há»£p cÃ¡c cá»™t láº¡i thÃ nh 1 chuá»—i duy nháº¥t
    df = df.with_columns(
        pl.concat_str(available_cols, separator="_").alias("flight_hash")
    )

    # 2. TÃ¬m Ä‘iá»ƒm cao nháº¥t cá»§a chuyáº¿n bay Ä‘Ã³ trong nhÃ³m
    df = df.with_columns(
        pl.max("pred_score")
        .over(["ranker_id", "flight_hash"])
        .alias("max_score_same_flight")
    )

    # 3. TÃ­nh Ä‘iá»ƒm pháº¡t (Penalty)
    # CÃ´ng thá»©c: Score_Má»›i = Score_CÅ© - Alpha * (Max_Score - Score_CÅ©)
    # Náº¿u lÃ  dÃ²ng tá»‘t nháº¥t: Score_Má»›i = Score_CÅ© (KhÃ´ng bá»‹ pháº¡t)
    # Náº¿u lÃ  dÃ²ng kÃ©m hÆ¡n: CÃ ng kÃ©m cÃ ng bá»‹ pháº¡t náº·ng
    df = df.with_columns(
        (
            pl.col("pred_score")
            - penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score"))
        ).alias("reorder_score")
    )

    # 4. Xáº¿p háº¡ng láº¡i dá»±a trÃªn Ä‘iá»ƒm má»›i
    df = df.with_columns(
        pl.col("reorder_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("new_selected")
    )

    return df.select(["Id", "ranker_id", "new_selected", "pred_score", "reorder_score"])

# ==============================================================================
# 3. THá»°C THI RE-RANK
# ==============================================================================
print("âš¡ Running Re-ranking Logic...")

# PENALTY_FACTOR: Há»‡ sá»‘ pháº¡t
# 0.1: Pháº¡t nháº¹ (Gáº§n nhÆ° giá»¯ nguyÃªn)
# 1.0 - 2.0: Pháº¡t máº¡nh (Ä�áº©y cÃ¡c báº£n sao xuá»‘ng dá»©t khoÃ¡t)
# KhuyÃªn dÃ¹ng: 0.5 Ä‘áº¿n 1.5
top_ranked = re_rank(pl_test_features, pl_submission, penalty_factor=0.2)

# ==============================================================================
# 4. Táº O FINAL SUBMISSION
# ==============================================================================
print("ğŸ“� Exporting Reranked Submission...")

final_sub = (
    pl_submission.join(top_ranked, on=["Id", "ranker_id"], how="left")
    .with_columns([
        pl.col("new_selected").alias("selected") # Ghi Ä‘Ã¨ rank cÅ© báº±ng rank má»›i
    ])
    .select(["Id", "ranker_id", "selected"])
    .sort(["ranker_id", "selected"])
)

# LÆ°u file
final_sub.write_csv("submission_reranked_postprocess.csv")
print("âœ… DONE! File: submission_reranked_postprocess.csv")

# 5. KIá»‚M TRA THAY Ä�á»”I (Sá»¬A Lá»–I)
# ==============================================================================
print("ğŸ“Š Calculating Stats...")

# 1. Join láº¡i báº£ng Ä‘iá»ƒm gá»‘c (pl_submission) vá»›i báº£ng rank má»›i (final_sub) theo Id
# Ä�á»ƒ Ä‘áº£m báº£o so sÃ¡nh Ä‘Ãºng dÃ²ng
compare_df = final_sub.join(pl_submission, on=["Id", "ranker_id"], how="left")

# 2. TÃ­nh láº¡i Rank gá»‘c tá»« Ä‘iá»ƒm sá»‘ cÅ© (trong ngá»¯ cáº£nh DataFrame)
compare_df = compare_df.with_columns(
    pl.col("pred_score")
    .rank(method="ordinal", descending=True)
    .over("ranker_id")
    .cast(pl.Int32)
    .alias("original_rank")
)

# 3. Ä�áº¿m sá»‘ dÃ²ng khÃ¡c biá»‡t
diff_count = compare_df.filter(
    pl.col("selected") != pl.col("original_rank")
).height

print(f"ğŸ“Š Re-ranking changed positions of {diff_count} rows.")
print("-" * 60)
print("âœ… QUY TRÃŒNH HOÃ€N Táº¤T! Báº¡n cÃ³ thá»ƒ ná»™p file 'submission_reranked_postprocess.csv'")


# import lightgbm as lgb
# import numpy as np
# import pandas as pd
# import gc
# import os
# import matplotlib.pyplot as plt

# print("ğŸš€ GIAI Ä�Oáº N 5: SINGLE LIGHTGBM (NATIVE API - GPU OPTIMIZED)")

# # ==============================================================================
# # 1. Cáº¤U HÃŒNH GPU & RAM (DÃ™NG CHO NATIVE API)
# # ==============================================================================
# # Ä�Ã¢y lÃ  cáº¥u hÃ¬nh 20% Stats (80% Train) tá»‘i Æ°u nháº¥t
# PARAMS_FINAL = {
#     "objective": "lambdarank",
#     "metric": "ndcg",
#     "eval_at": [3],
    
#     # --- GPU ---
#     "device": "gpu",
#     "gpu_platform_id": 0,
#     "gpu_device_id": 0,
    
#     # --- MEMORY SAFETY ---
#     "max_bin": 63,              # Giáº£m bin Ä‘á»ƒ nháº¹ VRAM
#     "two_round": True,          # Load dá»¯ liá»‡u 2 vÃ²ng (Chá»‘ng OOM)
    
#     # --- MODEL CAPACITY (255 LEAVES) ---
#     "num_leaves": 255,          # Ä�á»§ sÃ¢u cho 8 triá»‡u dÃ²ng dá»¯ liá»‡u
#     "min_data_in_leaf": 300,    # Chá»‘ng nhiá»…u
#     "learning_rate": 0.03,
#     "n_estimators": 3000,
    
#     # --- REGULARIZATION ---
#     "colsample_bytree": 0.8,
#     "subsample": 0.7,
#     "subsample_freq": 1,
#     "reg_lambda": 5.0,
    
#     "random_state": 42,
#     "n_jobs": -1,
#     "verbose": -1
# }

# # ==============================================================================
# # 2. CHUáº¨N Bá»Š FEATURES
# # ==============================================================================
# DROP_LIST = [
#     "ranker_id", "selected", "fold", "group_size", "size_tier", "Id", 
#     "requestDate", "flight_hash", "searchRoute",
#     "legs0_arrivalAt", "legs0_departureAt", "legs1_arrivalAt", "legs1_departureAt"
# ]

# # Lá»�c feature
# features = [c for c in pandas_df.columns if c not in DROP_LIST]

# # Lá»�c feature há»£p lá»‡ (Sá»‘ hoáº·c Category)
# valid_features = []
# for col in features:
#     dtype = pandas_df[col].dtype
#     if pd.api.types.is_numeric_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
#         valid_features.append(col)
# features = valid_features

# # XÃ¡c Ä‘á»‹nh Category
# cat_candidates = ["pricingInfo_isAccessTP", "frequentFlyer", "isVip", "bySelf", "sex", 
#                   "cancellation_status", "exchange_status", "has_corporate_tariff", 
#                   "has_access_tp", "is_vip_freq", "is_one_way", "has_fees", 
#                   "is_major_carrier", "is_direct_leg1"]
# categorical_features = [c for c in cat_candidates if c in features]

# print(f"ğŸ“‹ Final Features: {len(features)}")

# # ==============================================================================
# # 3. METRIC HITRATE (CHO NATIVE API)
# # ==============================================================================
# # Native API cáº§n hÃ m metric dáº¡ng closure Ä‘á»ƒ Ä‘Æ°a vÃ o feval
# import polars as pl
# def get_hitrate_eval(val_ids):
#     def eval_func(preds, eval_data):
#         y_true = eval_data.get_label()
#         # TÃ­nh toÃ¡n báº±ng Polars cho nhanh
#         df_tmp = pl.DataFrame({"y": y_true, "p": preds, "g": val_ids})
#         hits = df_tmp.sort("p", descending=True).group_by("g").head(3).filter(pl.col("y") == 1)
#         valid_g = df_tmp.group_by("g").len().filter(pl.col("len") > 0)
#         score = 0.0
#         if valid_g.height > 0:
#             score = hits.join(valid_g, on="g", how="inner").height / valid_g.height
#         return 'hitrate', score, True
#     return eval_func

# # ==============================================================================
# # 4. TRAINING LOOP (NATIVE API)
# # ==============================================================================
# final_test_preds = np.zeros(len(test_df_pd))
# N_SPLITS = 10
# all_history = [] # LÆ°u lá»‹ch sá»­ Ä‘á»ƒ váº½ biá»ƒu Ä‘á»“

# print(f"\nğŸ¥Š Báº¯t Ä‘áº§u Train {N_SPLITS} Folds (GPU Mode)...")

# for fold in range(N_SPLITS):
#     print(f"\n--- Fold {fold} ---")
    
#     # Chá»‰ láº¥y index (KhÃ´ng copy dá»¯ liá»‡u)
#     train_idx = pandas_df.index[pandas_df["fold"] != fold]
#     val_idx = pandas_df.index[pandas_df["fold"] == fold]
    
#     # Láº¥y ID cho metric
#     val_ranker_ids = pandas_df.loc[val_idx, "ranker_id"]
#     if isinstance(val_ranker_ids.dtype, pd.CategoricalDtype):
#         val_ranker_ids = val_ranker_ids.cat.codes.values
#     else:
#         val_ranker_ids = val_ranker_ids.values
    
#     # 1. Dataset Valid (Táº¡o trÆ°á»›c cho nháº¹)
#     dval = lgb.Dataset(
#         pandas_df.loc[val_idx, features],
#         label=pandas_df.loc[val_idx, "selected"],
#         group=pandas_df.loc[val_idx].groupby("ranker_id", sort=False).size().to_numpy(),
#         categorical_feature=categorical_features,
#         free_raw_data=True
#     )
    
#     # 2. Dataset Train
#     train_group = pandas_df.loc[train_idx].groupby("ranker_id", sort=False).size().to_numpy()
#     dtrain = lgb.Dataset(
#         pandas_df.loc[train_idx, features],
#         label=pandas_df.loc[train_idx, "selected"],
#         group=train_group,
#         categorical_feature=categorical_features,
#         free_raw_data=True
#     )
    
#     gc.collect()
    
#     # 3. Train
#     evals_result = {} # Há»©ng lá»‹ch sá»­ train
    
#     model = lgb.train(
#         PARAMS_FINAL,
#         dtrain,
#         num_boost_round=3000,
#         valid_sets=[dval],
#         valid_names=['valid'],
#          # <--- QUAN TRá»ŒNG: LÆ°u history
#         # feval=get_hitrate_eval(val_ranker_ids), # Báº­t dÃ²ng nÃ y náº¿u muá»‘n xem hitrate (sáº½ cháº­m hÆ¡n xÃ­u)
#         callbacks=[
#             lgb.early_stopping(100, verbose=False),
#             lgb.log_evaluation(500),
#             lgb.record_evaluation(evals_result)
#         ]
#     )
    
#     # LÆ°u history
#     all_history.append(evals_result['valid']['ndcg@3'])
    
#     # Log Score
#     best_score = model.best_score['valid']['ndcg@3']
#     print(f"   ğŸŒŸ Best NDCG@3: {best_score:.5f}")
    
#     # Predict (Batch)
#     final_test_preds += model.predict(test_df_pd[features]) / N_SPLITS
    
#     # Dá»�n dáº¹p
#     del dtrain, dval, model, train_group
#     gc.collect()

# # ==============================================================================
# # 5. Váº¼ BIá»‚U Ä�á»’ (CHO BÃ�O CÃ�O)
# # ==============================================================================
# print("\nğŸ“Š Váº½ biá»ƒu Ä‘á»“ Training History...")
# plt.figure(figsize=(10, 6))

# # Cáº¯t vá»� Ä‘á»™ dÃ i chung
# min_len = min([len(h) for h in all_history])
# trimmed = [h[:min_len] for h in all_history]
# avg_hist = np.mean(trimmed, axis=0)

# plt.plot(range(1, min_len+1), avg_hist, 'r-', linewidth=2, label='Average NDCG@3')
# for h in all_history:
#     plt.plot(range(1, len(h)+1), h, 'gray', alpha=0.3)

# plt.title(f'LightGBM Training History (GPU - 20% Stats)\nMean Best Score: {np.max(avg_hist):.5f}')
# plt.xlabel('Iterations')
# plt.ylabel('NDCG@3')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('training_chart.png')
# plt.show()

# # ==============================================================================
# # 6. XUáº¤T FILE
# # ==============================================================================
# print("\nğŸ“� Exporting Submission...")
# test_df_pd["score"] = final_test_preds
# test_df_pd["rank"] = test_df_pd.groupby("ranker_id")["score"].rank(method="first", ascending=False).astype(int)

# submission = test_df_pd[["Id", "ranker_id", "rank"]].copy()
# submission.rename(columns={"rank": "selected"}, inplace=True)

# data_dir = "../input/aeroclub-recsys-2025"
# if os.path.exists(f"{data_dir}/test.parquet"):
#     original = pd.read_parquet(f"{data_dir}/test.parquet", columns=["Id"])
#     original['Id'] = original['Id'].astype(str)
#     submission['Id'] = submission['Id'].astype(str)
#     final_sub = original.merge(submission, on="Id", how="left")
#     final_sub["selected"] = final_sub["selected"].fillna(1).astype(int)
#     final_sub.to_csv("submission_single_gpu_20stats.csv", index=False)
# else:
#     submission.to_csv("submission_single_gpu_20stats.csv", index=False)

# print("âœ… DONE ALL!")


# import lightgbm as lgb
# import numpy as np
# import pandas as pd
# import polars as pl
# import gc

# print("=" * 60)
# print("ğŸš€ GIAI Ä�Oáº N TRAIN: LOW RAM + CUSTOM HITRATE METRIC")
# print("=" * 60)

# target = "selected"

# # 1. Danh sÃ¡ch cÃ¡c cá»™t há»‡ thá»‘ng cháº¯c cháº¯n bá»�
# DROP_LIST = [
#     # â�Œ NHÃ“M THá»œI GIAN THÃ” (THá»¦ PHáº M CHÃ�NH)
#     "legs0_arrivalAt", "legs0_departureAt", 
#     "legs1_arrivalAt", "legs1_departureAt",
#     "requestDate", 
#     "legs0_segments0_departureFrom_airport_iata", # SÃ¢n bay cá»¥ thá»ƒ quÃ¡ cÅ©ng cÃ³ thá»ƒ gÃ¢y nhiá»…u
#     "legs0_segments0_arrivalTo_airport_iata",
    
#     # â�Œ CÃ�C Cá»˜T Há»† THá»�NG
#     "ranker_id", "selected", "fold", "group_size", "size_tier", "Id"
# ]

# # 2. Láº¥y táº¥t cáº£ cá»™t, trá»« cá»™t há»‡ thá»‘ng
# initial_features = [c for c in pandas_df.columns if c not in DROP_LIST]

# # 3. [FIX] Lá»�c bá»� cÃ¡c cá»™t cÃ³ kiá»ƒu dá»¯ liá»‡u khÃ´ng há»£p lá»‡ (Object, Datetime)
# # LightGBM chá»‰ cháº¥p nháº­n: number (int/float), bool, category
# valid_features = []
# print("âš™ï¸� Checking feature dtypes...")

# for col in initial_features:
#     dtype = pandas_df[col].dtype
#     # Giá»¯ láº¡i náº¿u lÃ  sá»‘, category hoáº·c bool
#     if pd.api.types.is_numeric_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(dtype):
#         valid_features.append(col)
#     else:
#         print(f"   â�Œ Dropping raw column: {col} ({dtype})")

# features = valid_features
# print(f"ğŸ“‹ Final Valid Features: {len(features)}")
# print(features)

# # XÃ¡c Ä‘á»‹nh láº¡i Categorical Features tá»« danh sÃ¡ch features sáº¡ch
# cat_cols = ["pricingInfo_isAccessTP", "frequentFlyer", "isVip", "bySelf", "sex", "cancellation_status", "exchange_status", "has_corporate_tariff", "has_access_tp", "is_vip_freq", "is_one_way", "has_fees", "is_major_carrier", "is_direct_leg1"]
# categorical_features = [c for c in cat_cols if c in features]

# # 1. HÃ€M METRIC (Cá»¦A Báº N - Ä�Ãƒ Tá»�I Æ¯U CHO NATIVE API)
# # ==============================================================================
# def hitrate_metric_polars(y_true, y_pred, group_ids):
#     """ HÃ m tÃ­nh toÃ¡n cá»‘t lÃµi báº±ng Polars """
#     try:
#         # Chuyá»ƒn group_ids sang numpy array náº¿u chÆ°a pháº£i
#         if hasattr(group_ids, 'values'): 
#             group_ids = group_ids.values
            
#         # Táº¡o Polars DataFrame (Zero-copy náº¿u cÃ³ thá»ƒ)
#         df_tmp = pl.DataFrame({
#             "y": y_true, 
#             "p": y_pred, 
#             "g": group_ids
#         })
        
#         # Logic tÃ­nh HitRate@3
#         # 1. Sort theo Ä‘iá»ƒm dá»± Ä‘oÃ¡n giáº£m dáº§n
#         # 2. Láº¥y Top 3 má»—i nhÃ³m
#         # 3. Kiá»ƒm tra xem cÃ³ vÃ© y=1 trong Ä‘Ã³ khÃ´ng
#         hits = df_tmp.sort("p", descending=True).group_by("g").head(3).filter(pl.col("y") == 1)
        
#         # Ä�áº¿m sá»‘ nhÃ³m há»£p lá»‡ (cÃ³ nhiá»�u hÆ¡n 0 vÃ©)
#         # LÆ°u Ã½: Logic cÅ© lÃ  >10, mÃ¬nh sá»­a thÃ nh >0 Ä‘á»ƒ bao quÃ¡t háº¿t, hoáº·c giá»¯ >10 tÃ¹y báº¡n
#         valid_g = df_tmp.group_by("g").len().filter(pl.col("len") > 0)
        
#         if valid_g.height == 0: 
#             return 0.0
            
#         return hits.join(valid_g, on="g", how="inner").height / valid_g.height
#     except Exception as e:
#         print(f"Metric Error: {e}")
#         return 0.0

# # ==============================================================================
# # 2. TRAINING LOOP
# # ==============================================================================
# final_test_preds = np.zeros(len(test_df_pd))
# N_SPLITS = 10

# # Cáº¥u hÃ¬nh Native (Ä�Ã£ tá»‘i Æ°u RAM)
# PARAMS_LOW_RAM = {
#     "objective": "lambdarank",
#     "metric": "ndcg",
#     "eval_at": [3],
#     "max_bin": 63,
#     "n_estimators": 3000,   
#     "learning_rate": 0.03,
#     "num_leaves": 127,
#     "min_data_in_leaf": 101, 
#     "reg_lambda": 2.0,
#     "colsample_bytree": 0.8,
#     "subsample": 0.7,
#     "subsample_freq": 1,
#     'lambdarank_truncation_level': 10,
#     "random_state": 42,
#     "n_jobs": -1,
#     "two_round": True,
#     "verbose": -1
# }

# print(f"\nğŸ¥Š Báº¯t Ä‘áº§u Train {N_SPLITS} Folds...")

# for fold in range(N_SPLITS):
#     print(f"\n--- Fold {fold} ---")
    
#     train_mask = pandas_df["fold"] != fold
#     val_mask = pandas_df["fold"] == fold
    
#     # --- CHUáº¨N Bá»Š ID CHO METRIC ---
#     # Native API khÃ´ng tá»± lÆ°u trá»¯ ranker_id, ta pháº£i truyá»�n nÃ³ vÃ o hÃ m metric
#     # Láº¥y ID cá»§a táº­p Valid vÃ  Ã©p kiá»ƒu an toÃ n
#     val_ranker_ids = pandas_df.loc[val_mask, "ranker_id"]
#     if isinstance(val_ranker_ids.dtype, pd.CategoricalDtype):
#         val_ranker_ids = val_ranker_ids.cat.codes.values # DÃ¹ng mÃ£ sá»‘ (int) cho nhanh
#     else:
#         val_ranker_ids = val_ranker_ids.values

#     # --- WRAPPER FUNCTION (Cáº¦U Ná»�I) ---
#     # HÃ m nÃ y káº¿t ná»‘i LightGBM Native vá»›i hÃ m Polars cá»§a báº¡n
#     def lgb_hitrate_eval(preds, eval_data):
#         # preds: Máº£ng Ä‘iá»ƒm dá»± Ä‘oÃ¡n tá»« model
#         # eval_data: Dataset chá»©a label thá»±c táº¿
#         y_true = eval_data.get_label()
        
#         # Gá»�i hÃ m tÃ­nh toÃ¡n Polars
#         # LÆ°u Ã½: val_ranker_ids Ä‘Æ°á»£c láº¥y tá»« scope bÃªn ngoÃ i (closure)
#         score = hitrate_metric_polars(y_true, preds, val_ranker_ids)
        
#         # Tráº£ vá»� format chuáº©n: (tÃªn, giÃ¡ trá»‹, cÃ ng_cao_cÃ ng_tá»‘t)
#         return 'hitrate', score, True

#     # --- Táº O DATASET ---
#     # Valid Set
#     dval = lgb.Dataset(
#         pandas_df.loc[val_mask, features], 
#         label=pandas_df.loc[val_mask, "selected"],
#         group=pandas_df.loc[val_mask].groupby("ranker_id", sort=False).size().to_numpy(),
#         categorical_feature=categorical_features,
#         free_raw_data=True
#     )
    
#     # Train Set
#     train_group = pandas_df.loc[train_mask].groupby("ranker_id", sort=False).size().to_numpy()
#     dtrain = lgb.Dataset(
#         pandas_df.loc[train_mask, features], 
#         label=pandas_df.loc[train_mask, "selected"],
#         group=train_group,
#         categorical_feature=categorical_features,
#         free_raw_data=True 
#     )
    
#     gc.collect()
    
#     # --- TRAIN Vá»šI CUSTOM METRIC ---
#     model = lgb.train(
#         PARAMS_LOW_RAM,
#         dtrain,
#         num_boost_round=2000,
#         valid_sets=[dval],
#         valid_names=['valid'],
#         # ğŸ”¥ Ä�Æ°a hÃ m metric vÃ o Ä‘Ã¢y
#         feval=lgb_hitrate_eval, 
#         callbacks=[
#             lgb.early_stopping(50, verbose=False),
#             lgb.log_evaluation(0) # Táº¯t log chi tiáº¿t
#         ]
#     )
    
#     # Log Káº¿t quáº£
#     # Láº¥y Ä‘iá»ƒm tá»‘t nháº¥t (lÆ°u Ã½ key lÃ  'hitrate')
#     best_score = model.best_score['valid']['hitrate']
#     print(f"   ğŸŒŸ Best HitRate Fold {fold}: {best_score:.4f}")
    
#     # Predict
#     final_test_preds += model.predict(test_df_pd[features]) / N_SPLITS
    
#     # Dá»�n dáº¹p
#     del dtrain, dval, model, train_group, val_ranker_ids
#     gc.collect()

# # ==============================================================================
# # 3. XUáº¤T FILE (GIá»® NGUYÃŠN)
# # ==============================================================================
# # ... (Pháº§n xuáº¥t file nhÆ° cÅ©)
# print("\nğŸ“� Exporting...")
# test_df_pd["score"] = final_test_preds
# test_df_pd["rank"] = test_df_pd.groupby("ranker_id")["score"].rank(method="first", ascending=False).astype(int)

# submission = test_df_pd[["Id", "ranker_id", "rank"]].copy()
# submission.rename(columns={"rank": "selected"}, inplace=True)

# data_dir = "../input/aeroclub-recsys-2025"
# if os.path.exists(f"{data_dir}/test.parquet"):
#     original = pd.read_parquet(f"{data_dir}/test.parquet", columns=["Id"])
#     original['Id'] = original['Id'].astype(str)
#     submission['Id'] = submission['Id'].astype(str)
    
#     final_sub = original.merge(submission, on="Id", how="left")
#     final_sub["selected"] = final_sub["selected"].fillna(1).astype(int)
#     final_sub.to_csv("submission_lowram_hitrate.csv", index=False)
# else:
#     submission.to_csv("submission_lowram_hitrate.csv", index=False)

# print("âœ… DONE! File: submission_lowram_hitrate.csv")






















