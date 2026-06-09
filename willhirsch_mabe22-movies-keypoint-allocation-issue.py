from matplotlib import pyplot as plt
import polars as pl
import seaborn as sns

def plot_frame_breakdown(video, frame):
    df = (pl.read_parquet(f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{video}.parquet')
            .filter(pl.col('video_frame') == frame)
            .with_columns(
                pl.col("bodypart").str.extract(r"(left|right|nose)$", 1).fill_null("center").alias("side")
            )
         )

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f'{video}/{frame}')
    sns.scatterplot(df, x='x', y='y', hue='mouse_id', ax=ax[0])
    sns.scatterplot(df, x='x', y='y', hue='side', hue_order = ['left','center','right','nose'], ax=ax[1])
    [a.set_aspect('equal') for a in ax]


plot_frame_breakdown('MABe22_movies/1100085542', 1100)


plot_frame_breakdown('AdaptableSnail/1212811043', 1100)


plot_frame_breakdown('InvincibleJellyfish/1106234543', 8500)


plot_frame_breakdown('MABe22_keypoints/1000217804', 1000)

