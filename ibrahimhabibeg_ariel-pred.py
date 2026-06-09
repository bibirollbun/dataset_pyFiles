from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
github_token = user_secrets.get_secret("GITHUB_TOKEN")
user = "ibrahimhabibeg"
CLONE_URL = f"https://oauth2:{github_token}@github.com/{user}/ariel-2025.git"
get_ipython().system(f"git clone {CLONE_URL}")


! cd ariel-2025 && uv build


! ls ariel-2025/dist


! pip install ariel-2025/dist/ariel_pred-0.0.1-py3-none-any.whl --target ./my-packages

