!pip install /kaggle/input/kprize-akami-deps/*.whl --force-reinstall --root-user-action ignore --no-deps --no-index --find-links /kaggle/input/kprize-akami-deps


import io
import os
import shutil
import sys
import kaggle_evaluation.konwinski_prize_inference_server


!cp -r /kaggle/input/kprize-akami-codes-kami codes


from codes.submit import predict_inner, REPO_PATH


instance_count = None


def get_number_of_instances(num_instances: int) -> None:
    """The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances


skip_prediction = False


def predict(
    problem_statement: str,
    repo_archive: io.BytesIO,
    pip_packages_archive: io.BytesIO,
    env_setup_cmds_templates: list[str],
) -> str:
    """
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    global skip_prediction

    with open("repo_archive.tar", "wb") as f:
        f.write(repo_archive.read())
    if os.path.exists(REPO_PATH):
        shutil.rmtree(REPO_PATH)
    shutil.unpack_archive("repo_archive.tar", extract_dir=REPO_PATH)
    os.remove("repo_archive.tar")

    patch_or_None = predict_inner(
        problem_statement,
        repo_archive,
        pip_packages_archive,
        env_setup_cmds_templates,
        skip_prediction,
        save_result=True,
        difficulty_threshold=0.5,
        output_dir="output/",
    )

    shutil.rmtree(REPO_PATH)

    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        skip_prediction = True

    return patch_or_None


inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances, predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/konwinski-prize/",  # Path to the entire competition dataset
            "/kaggle/tmp/konwinski-prize/",  # Path to a scratch directory for unpacking data.a_zip.
        ),
        use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
    )

