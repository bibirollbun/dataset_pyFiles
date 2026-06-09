!pip install webdataset


from pathlib import Path

import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class cfg:
    data_dir     = "/kaggle/input/waveform-inversion"
    output_dir   = "/kaggle/working"
    dataset_name = "fwi_dataset"
    stage        = "train"    # train|test
    target_dirs  = [
      "FlatVel_A",
      "FlatVel_B",
      "CurveVel_A",
      "CurveVel_B",
      "FlatFault_A",
      "FlatFault_B",
      # "CurveFault_A",
      # "CurveFault_B",
      # "Style_A",
      # "Style_B",
    ]
    maxsize         = 0.6e9    # 600MB (default: 3GB). Maximum size of each shard.
    num_used_shards = None
    test_size       = 0.2      # represent the proportion of the dataset to include in the test split
    batch_size      = 2
    seed            = 42
    debug           = True


def search_data_path(target_dirs, root_dir, shuffle=True):
    files = []
    for target_dir in target_dirs:
        data_dir = Path(root_dir, target_dir)
        assert data_dir.is_dir(), f"{data_dir} is not found"
        if Path(data_dir, "data").is_dir():
            in_files = sorted(Path(data_dir, "data").glob("*.npy"))
            out_files = sorted(Path(data_dir, "model").glob("*.npy"))
        else:
            in_files = sorted(data_dir.glob("seis*.npy"))
            out_files = sorted(data_dir.glob("vel*.npy"))
        assert len(in_files) == len(out_files)
        files += list(zip(in_files, out_files))
    if shuffle:
        np.random.shuffle(files)
    return files


def generate_sample(in_file, out_file=None):
    if out_file is None:  # test data
        seis = np.load(in_file)
        assert seis.shape == (5, 1000, 70)
        seis = seis.astype(np.float16)
        data = [{
            "__key__": in_file.stem,
            "sample_id.txt": in_file.stem,
            "seis.npy": seis,
        }]
    else:  # train/val data
        seis = np.load(in_file)
        assert seis.shape == (500, 5, 1000, 70)
        seis = seis.astype(np.float16)
        vel = np.load(out_file)
        assert vel.shape == (500, 1, 70, 70)
        vel = vel.astype(np.float16)
        for i, j in zip(in_file.parents, out_file.parents):
            if i == j:
                common_path = str(i)
                break
        data = []
        for i in range(len(seis)):
            sample_id = f"{in_file.stem}_{out_file.stem}_{i}"
            data.append({
                "__key__": common_path + "_" + sample_id,
                "sample_id.txt": sample_id,
                "seis.npy": seis[i],
                "vel.npy": vel[i],
            })
    return data


if cfg.stage == "train":
    data_paths = search_data_path(cfg.target_dirs, Path(cfg.data_dir, "train_samples"), shuffle=True)
else:
    data_paths = sorted(Path(cfg.data_dir, "test").glob("*.npy"))
    data_paths = [(i, None) for i in data_paths]
    if cfg.debug:
        data_paths = data_paths[:100]
data_paths[:3]


# !rm -fr /kaggle/working/*


dataset_name = f"{cfg.stage}_{cfg.dataset_name}"
output_dir = Path(cfg.output_dir, dataset_name)
output_dir.mkdir(parents=True, exist_ok=True)
output_dir


if len(list(output_dir.glob("*.tar"))) > 0:
    print(f"already exists tar files in {output_dir}, skip")
else:
    writer = wds.ShardWriter(str(Path(output_dir, "%04d.tar")), maxsize=cfg.maxsize)
    for in_file, out_file in tqdm(data_paths):
        data = generate_sample(in_file, out_file)
        for d in data:
            writer.write(d)
    writer.close()
data[0]


def get_shard_paths(root_dir, dataset_name, stage, num_shards=None, test_size=0.2, seed=42):
    assert stage in ["train", "test"]
    assert num_shards is None or num_shards > 1
    assert 0 < test_size < 1
    dataset_dir = Path(root_dir, f"{stage}_{dataset_name}")
    shard_paths = np.array(sorted(map(str, dataset_dir.glob("*.tar"))))
    if stage == "train":
        if num_shards is not None:
            rng = np.random.default_rng(seed)
            shard_paths = rng.choice(shard_paths, size=min(num_shards, len(shard_paths)), replace=False)
            shard_paths.sort()
        trn_idx, val_idx = train_test_split(np.arange(len(shard_paths)), test_size=test_size, random_state=seed, shuffle=True)
        trn_shard_paths = sorted(shard_paths[trn_idx])
        val_shard_paths = sorted(shard_paths[val_idx])
        print(f"# of train shards: {len(trn_shard_paths)}, # of val shards: {len(val_shard_paths)}")
        return trn_shard_paths, val_shard_paths
    else:
        print(f"# of test shards: {len(shard_paths)}")
        return sorted(shard_paths)


paths = get_shard_paths(
    cfg.output_dir,
    cfg.dataset_name,
    cfg.stage,
    num_shards=cfg.num_used_shards,
    test_size=cfg.test_size,
    seed=cfg.seed,
)
if cfg.stage == "train":
    train_paths, val_paths = paths
    print("example of train files:", train_paths[:3])
    print("example of val files:", val_paths[:3])
else:
    test_paths = paths
    print("example of test files:", test_paths[:3])


def get_dataset(paths, stage, seed=42):
    dataset = wds.WebDataset(paths, seed=seed).decode()
    if stage != "test":
        dataset = (
            dataset
            .to_tuple("sample_id.txt", "seis.npy", "vel.npy")
            .map(
                lambda x: {
                    "sample_id": x[0],
                    "seis": x[1],
                    "vel": x[2],
                }
            )
        )
        if stage == "train":
            dataset = dataset.shuffle(10)
    elif stage == "test":
        dataset = (
            dataset
            .to_tuple("sample_id.txt", "seis.npy")
            .map(
                lambda x: {
                    "sample_id": x[0],
                    "seis": x[1],
                }
            )
        )
    return dataset


if cfg.stage == "train":
    train_dataset = get_dataset(train_paths, "train", seed=cfg.seed)
    val_dataset = get_dataset(val_paths, "val", seed=cfg.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=1, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=1, drop_last=False, pin_memory=True)
else:
    test_dataset = get_dataset(test_paths, "test", seed=cfg.seed)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=1, drop_last=False, pin_memory=True)


if cfg.stage == "train":
    print("train dataloader")
    for batch in train_dataloader:
        print(batch)
        break
    print("val dataloader")
    for batch in val_dataloader:
        print(batch)
        break
else:
    print("test dataloader")
    for batch in test_dataloader:
        print(batch)
        break







