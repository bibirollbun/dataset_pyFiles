import polars as pl



df = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')


# Популярность авиалиний
df_airpop=df.lazy().group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected')).collect()
df_airpop_b=df.lazy().group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected')).collect()


df_airpop.columns=['airc', 'popty']
df_airpop_b.columns=['airc', 'popty2']
df_airpop=df_airpop.join(df_airpop_b, on='airc')


del df


test=pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')


#copy
sss=test.select(['Id', 'frequentFlyer', 'pricingInfo_isAccessTP', 'legs0_segments0_marketingCarrier_code',
 'legs0_segments1_marketingCarrier_code', 'legs0_segments2_marketingCarrier_code', 'legs0_segments3_marketingCarrier_code',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments2_marketingCarrier_code', 'legs1_segments3_marketingCarrier_code',
                 
                  'legs1_segments0_marketingCarrier_code', 'ranker_id'	, 'totalPrice' ])

#append
sss=sss.join(df_airpop, left_on='legs0_segments0_marketingCarrier_code', right_on='airc', how='left').drop('popty2')
sss=sss.join(df_airpop.drop('popty'), left_on='legs1_segments0_marketingCarrier_code', right_on='airc', how='left')


#legs count 
sss= sss.with_columns(
    pl.sum_horizontal(
        pl.col(col).is_not_null().cast(pl.UInt8) for col in ['legs0_segments0_marketingCarrier_code',
 'legs0_segments1_marketingCarrier_code', 'legs0_segments2_marketingCarrier_code', 'legs0_segments3_marketingCarrier_code', 
        'legs1_segments1_marketingCarrier_code', 'legs1_segments2_marketingCarrier_code', 'legs1_segments3_marketingCarrier_code',
                 
                  'legs1_segments0_marketingCarrier_code'
                                                            
                                                            
                                                            ]
    ).alias("l0_seg")
)


#handmade ml
sss=sss.sort(['ranker_id', 'pricingInfo_isAccessTP', 'l0_seg','popty', 'popty2', 'totalPrice' ], 
               descending=[True, True, False, True, True, False])


sss= sss.with_columns(
    pl.int_range(1, pl.len() + 1).alias("selected").over("ranker_id")      )

subm=test.select('Id', 'ranker_id' ).join(sss['Id', 'ranker_id', 'selected'], on=['Id', 'ranker_id'], how='left')


subm.write_csv('sumbission.csv')

