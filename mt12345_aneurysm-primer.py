import polars as pl
import altair as alt

df = pl.read_csv(
    '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv',
    schema_overrides={
        'PatientSex': pl.datatypes.Enum(["Male", "Female"]),
        'Modality': pl.datatypes.Enum(["MRA", "CTA", "MRI T2", "MRI T1post"])
    }).with_columns(total_aneurysm_locations=pl.sum_horizontal(pl.nth(range(4,17))))
aneurysms = df.filter(pl.col("Aneurysm Present") == 1)


full_dataset = df.select('PatientAge').to_series().plot.hist().properties(title="full dataset")
aneurysms_only = aneurysms.select('PatientAge').to_series().plot.hist(
    color=alt.value('orange')).properties(title="aneurysms only")

chart=(full_dataset|aneurysms_only).configure_title(fontSize=18, offset=15, orient='top', anchor='middle')
chart.title="Age distribution comparison"
chart


full_dataset_sex = df.group_by("PatientSex").len()
ratio_full_dataset = (
    full_dataset_sex.filter(pl.col("PatientSex")=="Female").select("len") /
    full_dataset_sex.filter(pl.col("PatientSex")=="Male").select("len")
).to_series()
aneurysms_only_sex = aneurysms.group_by("PatientSex").len()
ratio_aneuysms = (
    aneurysms_only_sex.filter(pl.col("PatientSex")=="Female").select("len") /
    aneurysms_only_sex.filter(pl.col("PatientSex")=="Male").select("len")
).to_series()

print("""Sex balance
Full dataset (F/M): {}
Aneurysms only (F/M): {}""".format(round(*ratio_full_dataset, 2), round(*ratio_aneuysms, 2)))


full_modality_counts = df.group_by('Modality').len().with_columns(
    (pl.col('len')/len(df)*100).round(2).alias("percent")
).sort('percent', descending=True)
full_modality_counts_chart = alt.Chart(full_modality_counts).mark_arc().encode(
    theta="len",
    color="Modality",
).properties(
    title=alt.TitleParams(
        text='Full dataset'
    )
)
aneurysm_only_modality_counts = aneurysms.group_by('Modality').len().with_columns(
    (pl.col('len')/len(aneurysms)*100).round(2).alias("percent")
).sort('percent', descending=True)
aneurysm_only_modality_counts_chart = alt.Chart(aneurysm_only_modality_counts).mark_arc().encode(
    theta="len",
    color="Modality",
).properties(
    title=alt.TitleParams(
        text='Aneurysms only')
)
chart = (full_modality_counts_chart| aneurysm_only_modality_counts_chart).properties(title=alt.TitleParams(
        text='Dataset composition',
        subtitle='by modality'))
chart = chart.configure_title(fontSize=18, offset=5, orient='top', anchor='middle')
chart


grouped_locations = aneurysms.with_columns(
    multiple_locations=(pl.col("total_aneurysm_locations")>1).cast(int),
    any_anterior=pl.any_horizontal(
        [
             'Left Infraclinoid Internal Carotid Artery',
             'Right Infraclinoid Internal Carotid Artery',
             'Left Supraclinoid Internal Carotid Artery',
             'Right Supraclinoid Internal Carotid Artery',
             'Left Middle Cerebral Artery',
             'Right Middle Cerebral Artery',
             'Anterior Communicating Artery',
             'Left Anterior Cerebral Artery',
             'Right Anterior Cerebral Artery',
        ]
    ).cast(int),
    any_aca_acoa_complex=pl.any_horizontal(
        [
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
        ]
                                          ).cast(int),
    any_ica=pl.any_horizontal(
        [
             'Left Infraclinoid Internal Carotid Artery',
             'Right Infraclinoid Internal Carotid Artery',
             'Left Supraclinoid Internal Carotid Artery',
             'Right Supraclinoid Internal Carotid Artery',
        ]
    ).cast(int),
    any_mca=pl.any_horizontal(
        [
             'Left Middle Cerebral Artery',
             'Right Middle Cerebral Artery',
        ]
    ).cast(int),
    any_posterior=pl.any_horizontal(
        [
             'Left Posterior Communicating Artery',
             'Right Posterior Communicating Artery',
             'Basilar Tip',
             'Other Posterior Circulation',
        ]
    ).cast(int)
)


grouped_locations.select(
    ['multiple_locations', 'any_anterior', 'any_aca_acoa_complex', 'any_ica', 'any_mca', 'any_posterior']
).sum() / len(grouped_locations)*100







