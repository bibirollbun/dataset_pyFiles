%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


PROJECT = "bigframes-dev" # replace with your project. 
# Refer to https://cloud.google.com/bigquery/docs/multimodal-data-dataframes-tutorial#required_roles for your required permissions

OUTPUT_BUCKET = "bigframes_blob_test" # replace with your GCS bucket. 
# The connection (or bigframes-default-connection of the project) must have read/write permission to the bucket. 
# Refer to https://cloud.google.com/bigquery/docs/multimodal-data-dataframes-tutorial#grant-permissions for setting up connection service account permissions.
# In this Notebook it uses bigframes-default-connection by default. You can also bring in your own connections in each method.

import bigframes
# Setup project
bigframes.options.bigquery.project = PROJECT

# Display options
bigframes.options.display.blob_display_width = 300
bigframes.options.display.progress_bar = None

import bigframes.pandas as bpd


# Create blob columns from wildcard path.
df_image = bpd.from_glob_path(
    "gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/*", name="image"
)
# Other ways are: from string uri column
# df = bpd.DataFrame({"uri": ["gs://<my_bucket>/<my_file_0>", "gs://<my_bucket>/<my_file_1>"]})
# df["blob_col"] = df["uri"].str.to_blob()

# From an existing object table
# df = bpd.read_gbq_object_table("<my_object_table>", name="blob_col")


# Take only the 5 images to deal with. Preview the content of the Mutimodal DataFrame
df_image = df_image.head(5)
df_image


# Combine unstructured data with structured data
df_image["author"] = ["alice", "bob", "bob", "alice", "bob"]  # type: ignore
df_image["content_type"] = df_image["image"].blob.content_type()
df_image["size"] = df_image["image"].blob.size()
df_image["updated"] = df_image["image"].blob.updated()
df_image


# filter images and display, you can also display audio and video types
df_image[df_image["author"] == "alice"]["image"].blob.display()


df_image["blurred"] = df_image["image"].blob.image_blur(
    (20, 20), dst=f"gs://{OUTPUT_BUCKET}/image_blur_transformed/", engine="opencv"
)
df_image["resized"] = df_image["image"].blob.image_resize(
    (300, 200), dst=f"gs://{OUTPUT_BUCKET}/image_resize_transformed/", engine="opencv"
)
df_image["normalized"] = df_image["image"].blob.image_normalize(
    alpha=50.0,
    beta=150.0,
    norm_type="minmax",
    dst=f"gs://{OUTPUT_BUCKET}/image_normalize_transformed/",
    engine="opencv",
)


# You can also chain functions together
df_image["blur_resized"] = df_image["blurred"].blob.image_resize((300, 200), dst=f"gs://{OUTPUT_BUCKET}/image_blur_resize_transformed/", engine="opencv")
df_image


from bigframes.ml import llm
gemini = llm.GeminiTextGenerator()


# Ask the same question on the images
df_image = df_image.head(2)
answer = gemini.predict(df_image, prompt=["what item is it?", df_image["image"]])
answer[["ml_generate_text_llm_result", "image"]]


# Ask different questions
df_image["question"] = ["what item is it?", "what color is the picture?"]


answer_alt = gemini.predict(df_image, prompt=[df_image["question"], df_image["image"]])
answer_alt[["ml_generate_text_llm_result", "image"]]


# Generate embeddings.
embed_model = llm.MultimodalEmbeddingGenerator()
embeddings = embed_model.predict(df_image["image"])
embeddings




