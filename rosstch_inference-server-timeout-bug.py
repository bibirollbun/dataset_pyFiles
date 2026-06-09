import io
import os
import kaggle_evaluation.konwinski_prize_inference_server
import concurrent.futures
import time


instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances

executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    try:
        fut = executor.submit(predict_)
        return fut.result(timeout=70)
    except Exception as e:
        print(f"Exception in predict_: {e}")
        return ""

SLEEP_INTERVAL = 5

def predict_():
    i = 0
    while True:
        time.sleep(SLEEP_INTERVAL)
        i += 1
        print(f"{i * SLEEP_INTERVAL} seconds passed")
    return "ok"


# Run inference

def main():
    inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
        get_number_of_instances,
        predict
    )

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
                '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
            ),
            use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Main throw exception: {e}")
        if executor:
            executor.shutdown(wait=False)

