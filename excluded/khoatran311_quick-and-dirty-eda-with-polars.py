import numpy as np
import pandas as pd
import polars as pl
import seaborn as sea
import polars.selectors as pls
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import warnings
warnings.filterwarnings("ignore")

def format_pl():
    """FLOAT DISPLAY FORMATTING"""
    pl.Config.set_fmt_float("mixed")
    """STRING FORMATTING"""
    pl.Config.set_fmt_str_lengths(50)
    """TABLE FORMATTING"""
    pl.Config.set_tbl_rows(8)
    pl.Config.set_tbl_cols(15)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cell_alignment("RIGHT")
    pl.Config.set_tbl_hide_dtype_separator(True)
    pl.Config.set_tbl_hide_column_data_types(True)

format_pl()


train = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")

train.head()


subset1_features = ["sequence_id", "sequence_type", "sequence_counter", "subject", "phase", "gesture", "orientation", "behavior"]
sub1 = train.select(
    pl.col(subset1_features)
)

sub1.head(4)


gesture_counts = sub1.group_by(
    pl.col("sequence_type"), pl.col("gesture")
).agg(
    pl.col("gesture").count().alias("count")
).sort(by="count", descending=True).to_pandas()

gesture_counts["pct"] = 100*gesture_counts["count"]/gesture_counts["count"].sum()

gesture_counts


plt.figure(figsize=(14, 6))

plt.subplot(121)
sea.barplot(
    data=gesture_counts,
    y="gesture",
    x="count",
    hue="sequence_type",
    dodge=False
)
plt.xticks(rotation=90)
plt.title("Frequency of Gestures in Data")
plt.xlabel("Frequency")
plt.ylabel("")

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
sea.barplot(
    data=gesture_counts,
    y="gesture",
    x="pct",
    hue="sequence_type",
    dodge=False
)
plt.xticks(rotation=90)
plt.title("Percentage of Gestures in Data")
plt.xlabel("Percentage (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()


sub1.group_by(
    pl.col("phase", "behavior")
).agg(
    pl.col("phase").count().alias("count")
).sort(by="phase", descending=True)


behavior_seq_table = sub1.group_by(
    pl.col("behavior", "sequence_type")
).agg(
    pl.col("behavior").count().alias("count")
).sort(by=("behavior","sequence_type"), descending=True)


orientation_seq_table = sub1.group_by(
    pl.col("orientation", "sequence_type")
).agg(
    pl.col("orientation").count().alias("count")
).sort(by=("orientation","sequence_type"), descending=True)


phase_seq_table = sub1.group_by(
    pl.col("phase", "sequence_type")
).agg(
    pl.col("phase").count().alias("count")
).sort(by=("phase","sequence_type"), descending=True)

print(behavior_seq_table)
print(orientation_seq_table)
print(phase_seq_table)


### Graphs table to analyze any relationships with sequence_type


"""Each subject contribute multiple kinds of sequences that are target/non-target"""

sub1.group_by(
    pl.col("subject", "sequence_id", "sequence_type")
).agg(
    pl.col("subject").count().alias("count")
).sort(
    by=("subject", "sequence_type")
)


"""What is the count of sequences per subject (how many sequences does each subject contributes) ?

Most subjects contributed 102 sequences, with 5 subjects contributing less, but that is not a large outlier issue. 
"""

## Sorts by subject 
subject_seq_table = sub1.group_by(
    pl.col("subject", "sequence_id",)
).agg(
    pl.col("subject").count().alias("count")
).sort(
    by=("subject")
)

## Counts subject
seq_per_subject_table = subject_seq_table.group_by(
    pl.col("subject")
).agg(
    pl.col("subject").count().alias("seq_per_subject_count")
)

seq_per_subject_table.group_by(
    pl.col("seq_per_subject_count")
).agg(
    pl.col("seq_per_subject_count").count().alias("count")
).sort(by="seq_per_subject_count", descending=True)


sub1.group_by(
    pl.col("subject", "sequence_id",)
).agg(
    pl.col("subject").count().alias("count")
).sort(
    by=("subject")
).group_by(
    pl.col("subject")
).agg(
    pl.col("subject").count().alias("seq_per_subject_count")
).filter(
    pl.col("seq_per_subject_count")!=102
)


"""Are the length distribution of target sequences different from non-target sequences?"""

seq_lengths = sub1.select(
    pl.col("sequence_id", "sequence_counter", "sequence_type")
).group_by(
    pl.col("sequence_id", "sequence_type")
).agg(
    pl.col("sequence_id").count().alias("length")
).to_pandas()

sea.histplot(x=np.log10(seq_lengths["length"]), hue=seq_lengths["sequence_type"])
plt.xlabel("Log10 Sequence Length")
plt.ylabel("Count")
plt.show()


target_lengths    = seq_lengths[seq_lengths["sequence_type"] == "Target"]["length"]
nontarget_lengths = seq_lengths[seq_lengths["sequence_type"] == "Non-Target"]["length"]

u_stat, p_value = mannwhitneyu(
    target_lengths, 
    nontarget_lengths,
    alternative="less"   ## median target < median nontarget
)

print(f"Ho: Median lengths of target and non-target sequences are not differnet")
print(f"Ha: Median length of target is less than median length of non-target sequences")
print(f"U = {u_stat:.3f}, p = {p_value:.4f}")


sea.kdeplot(x=target_lengths, label="Target", fill=True)
sea.kdeplot(nontarget_lengths, label="Non-Target", fill=True)
plt.title("Sequence Length Distributions")
plt.legend()
plt.show()


"""For each subject, what is the ratio of target/non-target sequences?"""

sub_seq_table = sub1.group_by(
    pl.col("subject", "sequence_id", "sequence_type")
).agg(
    pl.col("subject").count().alias("count")
).sort(
    by=("subject", "sequence_type")
).group_by(
    pl.col("subject", "sequence_type")
).agg(
    pl.col("subject").count().alias("count")
).sort(
    by=("subject", "sequence_type")
)


sub_seq_table = sub_seq_table.pivot(
    values="count",
    index="subject",
    columns="sequence_type"
).with_columns(
    (pl.col("Target")/pl.col("Non-Target")).alias("target_nonTarget_ratio")
)

ratio_counts = sub_seq_table.group_by(
    pl.col("target_nonTarget_ratio")
).agg(
    pl.col("target_nonTarget_ratio").count().alias("count")
).sort(by="count", descending=True)


outlier_ratios = sub_seq_table.filter(
    (pl.col("target_nonTarget_ratio") < 1.68) | 
    (pl.col("target_nonTarget_ratio") > 1.69)
)


print(sub_seq_table)
print(ratio_counts)
print(outlier_ratios)


sub2 = train.select(
    pl.col("^acc_.*$", "^rot_.*$", "^thm_.*$", "^tof_.*_v.*$")
)
sub2.head(4)


features_NA = sub2.select(
    pl.all().is_null().sum()
).unpivot(
    value_name="NA_count",
    variable_name="feature"
).filter(
    pl.col("NA_count")>0
).sort(
    by=("feature", "NA_count"),
    descending=True
).with_columns(
    (100*pl.col("NA_count")/len(sub2)).alias("NA_pct")
)

features_NA


sub2.select(
    pl.all().min()
).unpivot(
    value_name="min_value"
).sort(
    by="min_value"
)


sub2 = sub2.with_columns(
    # Acceleration magnitude
    pl.fold(
        acc = pl.lit(0),
        function = lambda acc,x: acc + x**2,
        exprs = pl.col("^acc_.*$")
    ).sqrt().alias("acc_mag"),

    # Rotational angle (rad) from neutral position
    (2*np.arccos(pl.col("rot_w").clip(-1,1))).alias("rot_angle"),
    
    # Thermopile mean
    (pl.fold(
        acc = pl.lit(0),
        function = lambda acc,x: acc + x,
        exprs = pl.col("^thm_.*$")
     )/5).alias("thm_mean"),
    
    # Thermopile std
    pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.std().alias("thm_std"),
    
    # Thermopile range: max - min
    (pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.max() - \
     pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.min()).alias("thm_range")
)


sensors_summaries = sub2.select(
    pl.col("acc_mag", "rot_angle", "thm_mean", "thm_std", "thm_range")
).drop_nulls().to_pandas()


plt.figure(figsize=(16,6))

plt.subplot(221)
sea.histplot(x=sensors_summaries["acc_mag"])
plt.xlabel("Acceleration Magnitude (m/sec^2)")
plt.title("Acceleration Magnitude Distribution")

plt.subplot(222)
sea.histplot(x=np.log10(sensors_summaries["acc_mag"]))
plt.xlabel("Log10 Acceleration Magnitude")
plt.title("Log Acceleration Magnitude Distribution")

plt.tight_layout()
plt.show()


sea.histplot(x=sensors_summaries["rot_angle"]*180/np.pi, kde=True)
plt.xlabel("Rotational Angle (Degrees)")
plt.title("Rotational Angles of Users")
plt.show()


k = 20000
sample_mean = sensors_summaries["thm_mean"].sample(n=k, random_state=42)
sample_std  = sensors_summaries["thm_std"].sample(n=k, random_state=42)
x_range = np.arange(0, k)

plt.figure(figsize=(16, 6))
plt.subplot(121)
sea.scatterplot(x=x_range,
                y=sample_mean, 
                alpha=.05)

plt.fill_between(x=x_range,
                 y1=sample_mean - sample_std,
                 y2=sample_mean + sample_std,
                 alpha=1,
                 label="Â±1 std")
plt.title("Mean Temperature over 5 Sensors Â±1 std")
plt.ylabel("Mean Temperature over 5 Sensors")
plt.xlabel("Sample")
plt.legend()

plt.subplot(122)
sea.histplot(x=sensors_summaries["thm_range"])
plt.title("Range of Temperatures over 5 Sensors (Max - Min)")
plt.xlabel("Temperature Range")


plt.tight_layout()
plt.show()


train2 = train.with_columns(
    # Acceleration magnitude
    pl.fold(
        acc = pl.lit(0),
        function = lambda acc,x: acc + x**2,
        exprs = pl.col("^acc_.*$")
    ).sqrt().alias("acc_mag"),

    # Rotational angle (rad) from neutral position
    (2*np.arccos(pl.col("rot_w").clip(-1,1))).alias("rot_angle"),
    
    # Thermopile mean
    (pl.fold(
        acc = pl.lit(0),
        function = lambda acc,x: acc + x,
        exprs = pl.col("^thm_.*$")
     )/5).alias("thm_mean"),
    
    # Thermopile std
    pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.std().alias("thm_std"),
    
    # Thermopile range: max - min
    (pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.max() - \
     pl.concat_list(["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]).list.min()).alias("thm_range")
)


train2.group_by("sequence_type").agg(
    pl.col("acc_mag").median().alias("median_acc_mag"),
    pl.col("rot_angle").median().alias("median_rot_angle"),
    pl.col("thm_mean").median().alias("median_thm"),
    pl.col("thm_std").median().alias("median_thm_std"),
    pl.col("thm_range").median().alias("median_thm_range"),
).sort(by="sequence_type")


train2.group_by("sequence_type").agg(
    pl.col("acc_mag").mean().alias("mean_acc_mag"),
    pl.col("rot_angle").mean().alias("mean_rot_angle"),
    pl.col("thm_mean").mean().alias("mean_thm"),
    pl.col("thm_std").mean().alias("mean_thm_std"),
    pl.col("thm_range").mean().alias("mean_thm_range"),
).sort(by="sequence_type")


train2.group_by(
    pl.col("sequence_type", "gesture")
).agg(
    pl.col("acc_mag").median().alias("median_acc_mag"),
    pl.col("rot_angle").median().alias("median_rot_angle"),
    pl.col("thm_mean").median().alias("median_thm"),
    pl.col("thm_std").median().alias("median_thm_std"),
    pl.col("thm_range").median().alias("median_thm_range"),
).sort(by="sequence_type").to_pandas()


k = 20000
sequence_data = train2.select(
    pl.col("sequence_type", "acc_mag", "rot_angle", "thm_mean", "thm_std", "thm_range")
).to_pandas()
sampled_data = sequence_data.sample(n=k, random_state=42)



plt.figure(figsize=(16, 12))

plt.subplot(3,2,1)
sea.boxplot(
    data=sampled_data,
    x="acc_mag",
    y="sequence_type",
)
plt.xlabel("Acceleration Magnitude (m/sec^2)")

plt.subplot(3,2,2)
sea.boxplot(
    data=sampled_data,
    x="rot_angle",
    y="sequence_type",
)
plt.xlabel("Rotational Angle (Radians)")

plt.subplot(3,2,3)
sea.boxplot(
    data=sampled_data,
    x="thm_mean",
    y="sequence_type",
)
plt.xlabel("Mean Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,4)
sea.boxplot(
    data=sampled_data,
    x="thm_std",
    y="sequence_type",
)
plt.xlabel("SD of Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,5)
sea.boxplot(
    data=sampled_data,
    x="thm_range",
    y="sequence_type",
)
plt.xlabel("Range of Temperature across 5 Sensors (Celsius)")

plt.tight_layout()
plt.show()


k = 20000

phase_orientation_sensors = train2.select(
    pl.col("phase", "orientation", "behavior", "acc_mag", "rot_angle", "thm_mean", "thm_std", "thm_range")
).to_pandas()

sampled_data = phase_orientation_sensors.sample(n=k, random_state=42)


plt.figure(figsize=(16, 12))

plt.subplot(3,2,1)
sea.boxplot(
    data=sampled_data,
    x="acc_mag",
    y="phase",
)
plt.xlabel("Acceleration Magnitude (m/sec^2)")

plt.subplot(3,2,2)
sea.boxplot(
    data=sampled_data,
    x="rot_angle",
    y="phase",
)
plt.xlabel("Rotational Angle (Radians)")

plt.subplot(3,2,3)
sea.boxplot(
    data=sampled_data,
    x="thm_mean",
    y="phase",
)
plt.xlabel("Mean Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,4)
sea.boxplot(
    data=sampled_data,
    x="thm_std",
    y="phase",
)
plt.xlabel("SD of Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,5)
sea.boxplot(
    data=sampled_data,
    x="thm_range",
    y="phase",
)
plt.xlabel("Range of Temperature across 5 Sensors (Celsius)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))

plt.subplot(3,2,1)
sea.boxplot(
    data=sampled_data,
    x="acc_mag",
    y="orientation",
)
plt.xlabel("Acceleration Magnitude (m/sec^2)")

plt.subplot(3,2,2)
sea.boxplot(
    data=sampled_data,
    x="rot_angle",
    y="orientation",
)
plt.xlabel("Rotational Angle (Radians)")

plt.subplot(3,2,3)
sea.boxplot(
    data=sampled_data,
    x="thm_mean",
    y="orientation",
)
plt.xlabel("Mean Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,4)
sea.boxplot(
    data=sampled_data,
    x="thm_std",
    y="orientation",
)
plt.xlabel("SD of Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,5)
sea.boxplot(
    data=sampled_data,
    x="thm_range",
    y="orientation",
)
plt.xlabel("Range of Temperature across 5 Sensors (Celsius)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))

plt.subplot(3,2,1)
sea.boxplot(
    data=sampled_data,
    x="acc_mag",
    y="behavior",
)
plt.xlabel("Acceleration Magnitude (m/sec^2)")

plt.subplot(3,2,2)
sea.boxplot(
    data=sampled_data,
    x="rot_angle",
    y="behavior",
)
plt.xlabel("Rotational Angle (Radians)")

plt.subplot(3,2,3)
sea.boxplot(
    data=sampled_data,
    x="thm_mean",
    y="behavior",
)
plt.xlabel("Mean Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,4)
sea.boxplot(
    data=sampled_data,
    x="thm_std",
    y="behavior",
)
plt.xlabel("SD of Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,5)
sea.boxplot(
    data=sampled_data,
    x="thm_range",
    y="behavior",
)
plt.xlabel("Range of Temperature across 5 Sensors (Celsius)")

plt.tight_layout()
plt.show()


k = 20000
sequence_data = train2.select(
    pl.col("gesture", "acc_mag", "rot_angle", "thm_mean", "thm_std", "thm_range")
).to_pandas()
sampled_data = sequence_data.sample(n=k, random_state=42)



plt.figure(figsize=(16, 12))

plt.subplot(3,2,1)
sea.boxplot(
    data=sampled_data,
    x="acc_mag",
    y="gesture",
)
plt.xlabel("Acceleration Magnitude (m/sec^2)")

plt.subplot(3,2,2)
sea.boxplot(
    data=sampled_data,
    x="rot_angle",
    y="gesture",
)
plt.xlabel("Rotational Angle (Radians)")

plt.subplot(3,2,3)
sea.boxplot(
    data=sampled_data,
    x="thm_mean",
    y="gesture",
)
plt.xlabel("Mean Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,4)
sea.boxplot(
    data=sampled_data,
    x="thm_std",
    y="gesture",
)
plt.xlabel("SD of Temperature across 5 Sensors (Celsius)")

plt.subplot(3,2,5)
sea.boxplot(
    data=sampled_data,
    x="thm_range",
    y="gesture",
)
plt.xlabel("Range of Temperature across 5 Sensors (Celsius)")

plt.tight_layout()
plt.show()

