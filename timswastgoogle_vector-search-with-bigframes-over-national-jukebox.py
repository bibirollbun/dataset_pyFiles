%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


import bigframes._config
import bigframes.pandas as bpd

bpd.options.bigquery.location = "US"

# Set to your GCP project ID.
bpd.options.bigquery.project = "swast-scratch"


df = bpd.read_json(
    "gs://cloud-samples-data/third-party/usa-loc-national-jukebox/jukebox.jsonl",
    engine="bigquery",
    orient="records",
    lines=True,
)


# Use `peek()` instead of `head()` to see arbitrary rows rather than the "first" rows.
df.peek()


df.shape


# For the purposes of a demo, select only a subset of rows.
df = df.sample(n=250)
df.cache()
df.shape


# As a side effect of how I extracted the song information from the HTML DOM,
# we ended up with lists in places where we only expect one item.
#
# We can "explode" to flatten these lists.
flattened = df.explode([
    "Recording Repository",
    "Recording Label",
    "Recording Take Number",
    "Recording Date",
    "Recording Matrix Number",
    "Recording Catalog Number",
    "Media Size",
    "Recording Location",
    "Summary",
    "Rights Advisory",
    "Title",
])
flattened.peek()


flattened.shape


flattened = flattened.assign(**{
    "GCS Prefix": "gs://cloud-samples-data/third-party/usa-loc-national-jukebox/",
    "GCS Stub": flattened['URL'].str.extract(r'/(jukebox-[0-9]+)/'),
})
flattened["GCS URI"] = flattened["GCS Prefix"] + flattened["GCS Stub"] + ".mp3"
flattened["GCS Blob"] = flattened["GCS URI"].str.to_blob()


flattened["Transcription"] = flattened["GCS Blob"].blob.audio_transcribe(
    model_name="gemini-2.0-flash-001",
    verbose=True,
)
flattened["Transcription"]


print(f"Successful rows: {(flattened['Transcription'].struct.field('status') == '').sum()}")
print(f"Failed rows: {(flattened['Transcription'].struct.field('status') != '').sum()}")
flattened.shape


# Show transcribed lyrics.
flattened["Transcription"].struct.field("content")


# Find all instrumentatal songs
instrumental = flattened[flattened["Transcription"].struct.field("content") == ""]
print(instrumental.shape)
song = instrumental.peek(1)
song


import gcsfs
import IPython.display

fs = gcsfs.GCSFileSystem(project='bigframes-dev')
with fs.open(song["GCS URI"].iloc[0]) as song_file:
    song_bytes = song_file.read()

IPython.display.Audio(song_bytes)


from bigframes.ml.llm import TextEmbeddingGenerator

text_model = TextEmbeddingGenerator(model_name="text-multilingual-embedding-002")


df_to_index = (
    flattened
    .assign(content=flattened["Transcription"].struct.field("content"))
    [flattened["Transcription"].struct.field("content") != ""]
)
embedding = text_model.predict(df_to_index)
embedding.peek(1)


# Check the status column to look for errors.
print(f"Successful rows: {(embedding['ml_generate_embedding_status'] == '').sum()}")
print(f"Failed rows: {(embedding['ml_generate_embedding_status'] != '').sum()}")
embedding.shape


embedding_table_id = f"{bpd.options.bigquery.project}.kaggle.national_jukebox"
embedding.to_gbq(embedding_table_id, if_exists="replace")


import bigframes.pandas as bpd

df_written = bpd.read_gbq(embedding_table_id)
df_written.peek(1)


from bigframes.ml.llm import TextEmbeddingGenerator

search_string = "walking home"

text_model = TextEmbeddingGenerator(model_name="text-multilingual-embedding-002")
search_df = bpd.DataFrame([search_string], columns=['search_string'])
search_embedding = text_model.predict(search_df)
search_embedding


import bigframes.bigquery as bbq

vector_search_results = bbq.vector_search(
    base_table=f"swast-scratch.scipy2025.national_jukebox",
    column_to_search="ml_generate_embedding_result",
    query=search_embedding,
    distance_type="COSINE",
    query_column_to_search="ml_generate_embedding_result",
    top_k=5,
)


vector_search_results.dtypes


results = vector_search_results[["Title", "Summary", "Names", "GCS URI", "Transcription", "distance"]].sort_values("distance").to_pandas()
results


print(results["Transcription"].struct.field("content").iloc[0])


import gcsfs
import IPython.display

fs = gcsfs.GCSFileSystem(project='bigframes-dev')
with fs.open(results["GCS URI"].iloc[0]) as song_file:
    song_bytes = song_file.read()

IPython.display.Audio(song_bytes)




