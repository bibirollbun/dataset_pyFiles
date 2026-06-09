import os
import pandas as pd
import polars as pl
import numpy as np
import kaggle_evaluation.jane_street_inference_server


class JaneStreetInferenceServer:
    def __init__(self):
        self.lags_ = None
    def predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        if lags is not None:
            self.lags_ = lags
        preprocessed_test = test.with_columns([
            pl.col('time_id').cast(pl.Float32) / pl.col('time_id').max(),
        ])
        predictions = test.select(
            'row_id', 
            pl.lit(0.0).alias('responder_6')
        )
        return predictions


def main():
    inference_server_handler = JaneStreetInferenceServer()
    inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(
        inference_server_handler.predict
    )
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
                '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
            )
        )


if __name__ == '__main__':
    main()

