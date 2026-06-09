# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ==============================================================================
# OMUXΩ∞KUT-OS OMUX004o Fine structure constant α × Collatz fusion beam
# Brain-enhanced version 1
# Authors: KANAMORI JUNKI OMUXΩ∞
# Assembled for ARC Prize 2025 AGI Contest
# Date: 2025-11-01
# Description: Full implementation with ARC Prize (Kaggle) Flat Submission Format compliance and enhanced sanitization.
# ==============================================================================
import sys
import os
import json
import time
import copy
import concurrent.futures
import multiprocessing
import traceback
import gc
import warnings
import signal
import glob
import hashlib
import random
import math
from typing import List, Dict, Tuple, Any, Callable, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict, deque
import importlib
from pathlib import Path
# Imports required for dynamic module generation
import threading
import subprocess
import shutil
import heapq

# ==============================================================================
# 0. 環境設定と初期化 (メインプロセス用)
# ==============================================================================
print("---OMUXΩ∞KUT-OS OMUX004o (Fine structure constant α & Collaze fusion Beam - ARC 2025) ---")

# ------------------------------------------------------------------------------
# Auto Resource Tuning (OMUX004o Core Feature - ARC 2025 12H Optimized)
# ------------------------------------------------------------------------------
def has_gpu():
    try:
        # nvidia-smiの存在とGPUの検出を試みる
        if shutil.which("nvidia-smi") is None:
            return False
        # [最適化] タイムアウトを設定して安全性を高める (nvidia-smiのハングアップ対策)
        out = subprocess.check_output(["nvidia-smi","-L"], stderr=subprocess.STDOUT, text=True, timeout=5)
        return "GPU" in out
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False

# 環境変数とnvidia-smiの結果を総合的に判断（Kaggle環境も考慮）
GPU = has_gpu() or (os.environ.get("CUDA_VISIBLE_DEVICES","") != "" or "KAGGLE_URL_BASE" in os.environ)

# ARC Prize 2025 ルール: 12時間 (43200秒)
TOTAL_TIME_LIMIT = 12 * 60 * 60
# 安全マージン（デフォルト30分）。環境変数で調整可能。
SAFETY_MARGIN = int(os.environ.get("ARC_SAFETY_MARGIN_S", 30 * 60))
# 約11.5時間を実行時間の上限とする（最低10分は確保）
GLOBAL_WALL_S = max(600, TOTAL_TIME_LIMIT - SAFETY_MARGIN)

# 基本パラメータ（CPU設定：安定性と精度追求のバランス）
CFG = {
    "MAX_BEAMS": 1000000000000,          # CPU時のビーム幅 (10→20)
    "MAX_CAND_PER_OP": 2,     # 候補生成上限 (4→8)
    "GEOM_TRIALS": 2,         # GeomMatchの試行回数 (2→8)
    "ENABLE_INDUCTION": True, # CPUではコストが高いため無効化
    "GLOBAL_WALL_S": GLOBAL_WALL_S,
    "TASK_SOFT_CAP_S": 180,   # 1タスクあたりの目安時間 (90s→180s/3分)
}

if GPU:
    # GPUモード設定：12時間を見据えた積極的な探索設定
    CFG.update({
        "MAX_BEAMS": 1000000000000,             # 18→38 (探索空間を大幅拡大。Governorによる調整を前提)
        "MAX_CAND_PER_OP": 2,       # 5→6
        "GEOM_TRIALS": 2,           # 4→6
        "ENABLE_INDUCTION": True,    # GPUの計算力で探索幅を稼ぐ
        "TASK_SOFT_CAP_S": 600,      # 1タスク目安時間 (240s→600s/10分)。難問への対応力を最大化。
    })
    # CuPy/Numba系を使う実装がある場合
    os.environ.pop("CUPY_DISABLED", None)

# 環境変数による上書き（デバッグ・緊急調整用）
CFG["MAX_BEAMS"] = int(os.environ.get("ARC_MAX_BEAMS", CFG["MAX_BEAMS"]))
CFG["TASK_SOFT_CAP_S"] = int(os.environ.get("ARC_TASK_CAP_S", CFG["TASK_SOFT_CAP_S"]))

print(f"[AutoTune] ARC2025 Optimized (Limit: {TOTAL_TIME_LIMIT/3600:.2f}h, Margin: {SAFETY_MARGIN/60:.1f}m, Wall: {GLOBAL_WALL_S/3600:.2f}h)")
print(f"[AutoTune] GPU={GPU} | Beams={CFG['MAX_BEAMS']} | Cand/Op={CFG['MAX_CAND_PER_OP']} | "
      f"GeomTrials={CFG['GEOM_TRIALS']} | Induction={CFG['ENABLE_INDUCTION']} | TaskCap={CFG['TASK_SOFT_CAP_S']}s")

# ------------------------------------------------------------------------------
# 環境検出とディレクトリ設定
# ------------------------------------------------------------------------------
# マルチプロセッシングの設定 ('spawn'の適用を試みる。GPU利用時や一貫性確保に推奨)
MP_START_METHOD = 'spawn'
try:
    # メインプロセスでのみ実行可能
    if __name__ == "__main__":
        if MP_START_METHOD in multiprocessing.get_all_start_methods():
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method is None:
                # force=Falseで安全に設定を試みる
                multiprocessing.set_start_method(MP_START_METHOD, force=False)
                print(f"INFO: Multiprocessing start method set to '{MP_START_METHOD}'.")
            elif current_method != MP_START_METHOD:
                 # すでに開始されている場合は変更不可
                 print(f"INFO: Multiprocessing already started with '{current_method}'. Cannot change to '{MP_START_METHOD}'.")
            else:
                 print(f"INFO: Multiprocessing already using '{MP_START_METHOD}'.")
        else:
            print(f"INFO: '{MP_START_METHOD}' not available. Using default method.")
except RuntimeError as e:
    print(f"INFO: Could not set start method (RuntimeError). Using: {multiprocessing.get_start_method()}. Error: {e}")
except Exception as e:
    print(f"WARNING: Failed to configure multiprocessing: {e}")


KAGGLE_ENV = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if KAGGLE_ENV:
    print("Running in Kaggle environment.")
    INPUT_DIR = Path("/kaggle/input")
    WORKING_DIR = Path("/kaggle/working")
else:
    print("Running in Local environment.")
    # ローカル環境のデフォルトパス設定
    INPUT_DIR = Path("./evaluation_data")
    WORKING_DIR = Path("./working")
    INPUT_DIR.mkdir(exist_ok=True, parents=True)

WORKING_DIR.mkdir(exist_ok=True, parents=True)

# モジュールをインポート可能にするため、WORKING_DIRをパスに追加
if str(WORKING_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR.resolve()))

# ==============================================================================
# 1. gpu_distribute.py の動的生成 (Robust Distributed Executor)
# ==============================================================================
# (OMUX004o: CPU並列実行効率の改善、スレッド制限強化、ロギング改善、安定性向上)
GPU_DISTRIBUTE_CODE = r'''
import os, time, traceback, multiprocessing as mp
import sys
import gc
import shutil
import random

# 互換性レイヤー
try:
    from queue import Empty as QueueEmpty, Full as QueueFull
except ImportError:
    from Queue import Empty as QueueEmpty, Full as QueueFull

try:
    TimeoutError
except NameError:
    TimeoutError = OSError

def _worker_init(gpu_id:int):
    # ワーカーごとの環境設定：GPU固定とスレッド数制限
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # CPUモード
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # 数値計算ライブラリのスレッド数を1に制限（プロセス並列の効果を最大化）
    # [最適化] 関連する主要な環境変数をすべて設定し、CPU競合を防ぐ
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Initialize random seed unique to this worker process
    try:
        seed = (os.getpid() ^ int(time.time() * 1000)) & 0xffffffff
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
    except Exception:
        pass

# ワーカープロセスのメインループ
def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue, gpu_id:int, solve_one):
    _worker_init(gpu_id)

    while True:
        try:
            # タスクの取得（タイムアウト付きで待機し、無駄なCPU消費を抑制）
            item = task_queue.get(timeout=10.0)
        except QueueEmpty:
            continue

        if item is None:
            # 終了シグナル
            break

        task_id, task_blob = item
        res = None
        try:
            # タスク実行
            res = solve_one(task_blob)
            result_queue.put((task_id, res, None))
        except Exception as e:
            # エラーハンドリング
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"[Worker {os.getpid()} GPU:{gpu_id}] Error processing {task_id}: {err_msg}", file=sys.stderr)
            # エラー時も結果を返す
            result_queue.put((task_id, None, err_msg))
        finally:
            # メモリクリーンアップ（重要）
            if res is not None: del res
            if task_blob is not None: del task_blob
            gc.collect()
            # GPUメモリ解放（もしCuPy等を使っている場合）
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except ImportError:
                pass
            except Exception:
                pass

# 分散実行のメイン関数
def distribute(tasks:list, solve_one, gpu_ids=None, max_inflight_per_gpu=1, timeout=None):

    # GPU IDの自動検出ロジック
    if gpu_ids is None:
        # 1. nvidia-smiによる検出
        try:
            if shutil.which("nvidia-smi") is not None:
                import subprocess
                # タイムアウトを設定して安全性を確保
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode("utf-8").strip().splitlines()
                gpu_ids = [int(i.strip()) for i in out if i.strip().isdigit()]
            else:
                gpu_ids = []
        except Exception: # subprocess.TimeoutExpiredも含む
            gpu_ids = []

        # 2. CPUモード（フォールバックと効率改善）
        if not gpu_ids:
            cpu_count = mp.cpu_count() or 2
            # [最適化] CPU並列実行効率の改善: コア数の半分を使用し、最大8並列に制限（メモリ安全と効率のため）
            MAX_CPU_WORKERS = 8
            n_workers = max(1, min(MAX_CPU_WORKERS, cpu_count // 2))
            gpu_ids = [-1] * n_workers # -1はCPUを意味する

    print(f"[Distribute] Starting distribution across {len(gpu_ids)} Workers (IDs: {gpu_ids})")

    # マルチプロセッシングコンテキストの取得 (メインプロセスで設定されたメソッドを利用)
    ctx = mp.get_context()

    # キューの設定
    queue_size = max_inflight_per_gpu * len(gpu_ids)
    tq = ctx.Queue(maxsize=queue_size)
    rq = ctx.Queue()

    # ワーカープロセスの起動
    procs = []
    for gid in gpu_ids:
        # daemon=Trueでメインプロセス終了時に自動終了するように設定
        p = ctx.Process(target=_worker_loop, args=(tq, rq, gid, solve_one), daemon=True)
        p.start(); procs.append(p)

    # タスク投入と結果収集のメインループ
    task_iter = iter(tasks)
    tasks_in_flight = 0
    results = {}
    start = time.time()
    need = len(tasks)
    stop_signals_sent = False

    while len(results) < need:
        # 1. 新規タスクの投入
        if not stop_signals_sent:
            while tasks_in_flight < queue_size:
                try:
                    next_task = next(task_iter)
                    # ブロックせずに投入を試みる
                    tq.put(next_task, block=False)
                    tasks_in_flight += 1
                except StopIteration:
                    # 全タスク投入完了 -> 終了シグナル送信
                    print(f"[Distribute] All tasks submitted. Sending stop signals to workers.")
                    for _ in procs:
                        try:
                            # 確実にシグナルが届くようにタイムアウト付きでブロッキング投入を試みる
                            tq.put(None, block=True, timeout=5.0)
                        except QueueFull:
                             print(f"[Distribute] Warning: Task queue full when sending stop signal.", file=sys.stderr)
                        except Exception: pass
                    stop_signals_sent = True
                    break
                except QueueFull:
                    # キューがいっぱいなら投入を中断し、結果収集へ移る
                    break
                except Exception as e:
                    print(f"[Distribute] Error putting task to queue: {e}", file=sys.stderr)
                    break

        # 2. 結果の収集
        try:
            # 1秒待機して結果を取得
            tid, res, err = rq.get(timeout=1.0)
            results[tid] = (res, err)
            tasks_in_flight -= 1

            # [最適化] 進捗ログ（詳細表示：ETAとレート）
            # 頻繁に出力しすぎないように調整
            if len(results) % max(1, len(gpu_ids)) == 0 or len(results) == need:
                elapsed = time.time() - start
                rate = len(results) / elapsed if elapsed > 0 else 0
                eta_s = (need - len(results)) / rate if rate > 0 else 0
                # ETAを時間：分：秒形式で表示
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_s))
                elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                print(f"[Distribute] Progress: {len(results)}/{need} | Elapsed: {elapsed_str} | Rate: {rate:.2f} t/s | ETA: {eta_str}")

        except QueueEmpty:
            # 結果待ち状態

            # タイムアウトチェック
            if timeout and (time.time() - start) > timeout:
                print(f"[Distribute] Overall timeout ({timeout}s) reached. Stopping.", file=sys.stderr)
                break

            # ワーカー全滅チェック（ロバスト性向上）
            if not any(p.is_alive() for p in procs) and (tasks_in_flight > 0 or (not stop_signals_sent and len(results) < need)):
                # タスクが残っているのにワーカーがいない場合は異常事態
                print(f"[Distribute] CRITICAL: All workers died unexpectedly. Tasks in flight: {tasks_in_flight}.", file=sys.stderr)
                break

    # 終了処理
    # タイムアウト等で中断した場合、残りのワーカーに終了シグナルを送信
    if not stop_signals_sent:
        print(f"[Distribute] Sending final stop signals due to interruption.")
        for _ in procs:
            try:
                # 念のためブロッキングで送信
                tq.put(None, block=True, timeout=5.0)
            except: pass

    # プロセスの終了待機とクリーンアップ（タイムアウト延長とログ追加）
    print(f"[Distribute] Waiting for workers to terminate...")
    for i, p in enumerate(procs):
        # [最適化] 各プロセス最大120秒待機（長めのタスクが完了するのを安全に待つ）
        p.join(timeout=120.0)
        if p.is_alive():
            print(f"[Distribute] Worker {i} (PID: {p.pid}) did not terminate gracefully. Forcing termination.", file=sys.stderr)
            p.terminate() # 強制終了
            p.join(timeout=10.0)

        # プロセスリソースの解放 (Python 3.7+)
        if hasattr(p, 'close'):
            try: p.close()
            except: pass

    # キューのクローズ
    try: tq.close(); rq.close()
    except: pass

    print(f"[Distribute] Distribution finished. Total time: {time.time() - start:.1f}s. Completed: {len(results)}/{need}.")
    return results
'''
# ==============================================================================
# 2. kutos_worker.py の動的生成 (Sextuple Beam AI Logic)
# ==============================================================================
# (OMUX004o: MemGuard強化、Governor積極性調整、依存関係ハンドリング改善)

# メインプロセスで決定したCFG設定をワーカーコードに注入
WORKER_CONFIG_INJECTION = f'''
# [AutoTune Injection] メインプロセスから設定を継承
GPU_ENABLED = {GPU}
CFG_INJECTED = {CFG}
'''

WORKER_MODULE_CODE = r'''
import sys
import os
import json
import time
import copy
import concurrent.futures
import traceback
import gc
import warnings
import signal
import hashlib
import random
import math
import threading
import subprocess
import shutil
import statistics
import heapq
from typing import List, Dict, Tuple, Any, Callable, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict, deque

# ==============================================================================
# 0. Imports and Global Setup (ワーカープロセス用)
# ==============================================================================

# === AutoTune設定の適用 ===
''' + WORKER_CONFIG_INJECTION + r'''
# =========================

# ------------------------------------------------------------------------------
# 依存ライブラリのインポートとフォールバック
# ------------------------------------------------------------------------------

# NumPyのインポートとDummyNp（フォールバック）
class DummyNp:
    # [最適化] NumPyがない場合のスタブ実装。エラー抑制を強化。
    def array(self, data, *args, **kwargs): return data
    def ndarray(self, *args, **kwargs): return None
    def issubdtype(self, *args, **kwargs): return False
    def integer(self, *args, **kwargs): return int
    def uint8(self, *args, **kwargs): return int
    def bool(self, *args, **kwargs): return bool
    def float32(self, *args, **kwargs): return float

    # 基本的な集約・論理演算スタブ
    def count_nonzero(self, arr, *args, **kwargs):
        try: return sum(1 for x in self.flatten(arr) if x != 0)
        except: return 0

    def sum(self, arr, *args, **kwargs):
        try: return sum(self.flatten(arr))
        except: return 0

    def mean(self, arr, *args, **kwargs):
        try:
            flat = self.flatten(arr)
            return sum(flat) / len(flat) if flat else 0
        except: return 0

    def max(self, arr, *args, **kwargs):
        try: return max(self.flatten(arr))
        except: return 0

    def min(self, arr, *args, **kwargs):
        try: return min(self.flatten(arr))
        except: return 0

    def logical_and(self, a, b): return a and b
    def logical_or(self, a, b): return a or b
    def maximum(self, a, b): return max(a, b) if hasattr(a, '__gt__') else a
    def minimum(self, a, b): return min(a, b) if hasattr(a, '__lt__') else a

    def unique(self, arr, return_counts=False, *args, **kwargs):
        flat = self.flatten(arr)
        counts = Counter(flat)
        if return_counts: return list(counts.keys()), list(counts.values())
        return list(counts.keys())

    def flatten(self, arr):
        if isinstance(arr, list):
            try:
                # 2Dリストのフラット化を試みる
                return [x for row in arr for x in row]
            except TypeError:
                # 1Dリストまたはその他のイテラブル
                try: return list(arr)
                except TypeError: return []
        return []

    # 幾何学変換のリストベース実装
    def rot90(self, m, k=1):
        if not isinstance(m, list) or not m: return m
        try:
            if k % 4 == 1: return list(map(list, zip(*m[::-1])))
            elif k % 4 == 2: return [row[::-1] for row in m[::-1]]
            elif k % 4 == 3: return list(map(list, zip(*m)))[::-1]
            return m
        except: return m
    def fliplr(self, m): return [row[::-1] for row in m] if isinstance(m, list) else m
    def flipud(self, m): return m[::-1] if isinstance(m, list) else m
    def transpose(self, m):
        if isinstance(m, list) and m:
            try: return list(map(list, zip(*m)))
            except: return m
        return m
    def argwhere(self, m):
        if isinstance(m, list) and m:
            return [(i, j) for i, row in enumerate(m) for j, val in enumerate(row) if val]
        return []
    def zeros_like(self, m):
        if isinstance(m, list) and m:
            try: return [[0]*len(m[0]) for _ in range(len(m))]
            except IndexError: return []
        return []
    def array_equal(self, a, b): return a == b

    # その他のアクセスに対するフォールバック
    def __getattr__(self, name):
        # ログ出力を抑制し、静かに失敗させる
        return lambda *args, **kwargs: None

try:
    import numpy as np
    # NumPy利用時の設定（推奨）
    np.seterr(all='ignore')
except ImportError:
    np = DummyNp()
    print(f"[Worker {os.getpid()}] CRITICAL: NumPy not found. Falling back to DummyNp. Performance will be severely degraded.", file=sys.stderr)
except Exception as e:
    # インポート以外のエラー（例：C拡張のロード失敗）
    np = DummyNp()
    print(f"[Worker {os.getpid()}] ERROR: NumPy initialization failed: {e}. Falling back to DummyNp.", file=sys.stderr)


# SciPyのインポート（オブジェクト抽出と幾何学処理に重要）
scipy_label = scipy_find_objects = scipy_binary_fill_holes = None
try:
    from scipy.ndimage import label as scipy_label, find_objects as scipy_find_objects, binary_fill_holes as scipy_binary_fill_holes
except ImportError:
    # 古いSciPyバージョンへのフォールバック
    try:
        from scipy.ndimage.measurements import label as scipy_label, find_objects as scipy_find_objects
        from scipy.ndimage.morphology import binary_fill_holes as scipy_binary_fill_holes
    except ImportError:
        pass

# SciPyの可用性確認ログ
if not (scipy_label and scipy_find_objects and scipy_binary_fill_holes):
    print(f"[Worker {os.getpid()}] WARNING: Scipy import failed or incomplete. Object detection logic will be unavailable or degraded.", file=sys.stderr)

# psutil (メモリ監視用)
try:
    import psutil
except ImportError:
    psutil = None
    # Linux環境ではフォールバックがあるためINFOレベルとする
    if sys.platform.startswith("linux"):
        print(f"[Worker {os.getpid()}] INFO: psutil not found. Memory monitoring will rely on OS fallback (/proc/meminfo).", file=sys.stderr)
    else:
        print(f"[Worker {os.getpid()}] WARNING: psutil not found. Memory monitoring disabled.", file=sys.stderr)


warnings.filterwarnings("ignore")

# ==============================================================================
# 0.5. omux_memguard.py (ResourceMonitor & BeamGovernor)
# ==============================================================================
# (OMUX004o: 監視の堅牢性向上、Governorの積極性調整)

# リソース読み取りヘルパー関数

def _read_ram() -> Tuple[int, int]:
    # [最適化] psutilを優先し、利用可能メモリ(available)を基に使用量を計算する
    if psutil:
        try:
            vm = psutil.virtual_memory()
            # vm.availableはシステムが利用可能なメモリ量を正確に示す
            used = vm.total - vm.available
            return int(used), int(vm.total)
        except Exception:
            pass # フォールバックへ

    # [最適化] Linux系OSでのフォールバック実装 (/proc/meminfo)
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.readlines()
            
            mem_total = None
            mem_available = None
            
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024 # KB to Bytes
                elif line.startswith("MemAvailable:"):
                    # MemAvailable (カーネル3.14以降) を優先的に使用
                    mem_available = int(line.split()[1]) * 1024 # KB to Bytes
            
            if mem_total is not None and mem_available is not None:
                mem_used = mem_total - mem_available
                return int(mem_used), int(mem_total)

        except Exception:
            pass

    # 最終フォールバック (監視不可。現実的な値として128GBを仮定)
    return 0, 128 * 1024 * 1024 * 1024

def _read_disk(path: str = "/") -> Tuple[int, int]:
    try:
        if 'shutil' in globals() and hasattr(shutil, 'disk_usage'):
            # パスが存在しない場合はルートを参照
            if not os.path.exists(path): path = "/"
            du = shutil.disk_usage(path)
            return du.used, du.total
    except Exception: pass
    return 0, 0

def _read_gpus() -> List[Tuple[int, int, int]]:
    # nvidia-smiを使用してGPUメモリ使用状況を取得
    # [最適化] タイムアウトを設定し、コマンドのハングアップから保護する
    try:
        if 'subprocess' not in globals() or 'shutil' not in globals() or shutil.which("nvidia-smi") is None: return []

        # 5秒のタイムアウトを設定。エンコーディングを明示。
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5, encoding="utf-8"
        ).strip().splitlines()

        res = []
        for ln in out:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                try:
                    idx = int(parts[0]); used_mb = int(parts[1]); tot_mb = int(parts[2])
                    # MB単位をバイト単位に変換して統一
                    res.append((idx, used_mb * 1024*1024, tot_mb * 1024*1024))
                except ValueError: continue
        return sorted(res, key=lambda x: x[0])
    except subprocess.TimeoutExpired:
        print(f"[MemGuard] WARNING: nvidia-smi command timed out.", file=sys.stderr)
        return []
    except Exception:
        return []

class ResourceSnapshot:
    __slots__ = ("t","ram_used","ram_total","disk_used","disk_total","gpus")
    def __init__(self, t:float, ram_used:int, ram_total:int, disk_used:int, disk_total:int, gpus:List[Tuple[int,int,int]]):
        self.t=t; self.ram_used=ram_used; self.ram_total=ram_total
        self.disk_used=disk_used; self.disk_total=disk_total; self.gpus=gpus

    def ram_frac(self) -> float:
        # バイト単位での計算
        return (self.ram_used / max(self.ram_total,1.0)) if self.ram_total else 0.0

    def vram_fracs(self) -> List[float]:
        # gpus: (idx, used_bytes, total_bytes)
        return [(u / max(t,1.0)) if t else 0.0 for _,u,t in self.gpus]

class ResourceMonitor:
    # バックグラウンドスレッドでリソース使用状況を監視
    def __init__(self, sample_sec:float=1.0, disk_path:str="/kaggle/working"):
        self._thr = None
        if 'threading' not in globals() or threading.Thread is None:
            print(f"[MemGuard] INFO: Threading not available. ResourceMonitor disabled.", file=sys.stderr)
            return

        self.sample_sec = sample_sec
        self.disk_path = disk_path
        # 初期スナップショットを取得
        self._snap = self._capture_snapshot()
        self._stop = threading.Event()

    def _capture_snapshot(self) -> ResourceSnapshot:
        t = time.time()
        ru, rt = _read_ram(); du, dt = _read_disk(self.disk_path); gpus = _read_gpus()
        return ResourceSnapshot(t, ru, rt, du, dt, gpus)

    def start(self) -> "ResourceMonitor":
        if self._thr is None and hasattr(self, '_stop'):
            self._stop.clear()
            # daemon=Trueでメインスレッド終了時に自動終了
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
        return self

    def stop(self):
        if self._thr and hasattr(self, '_stop'):
            self._stop.set()
            # 停止まで最大5秒待機
            self._thr.join(timeout=5.0)
            self._thr = None

    def snapshot(self) -> ResourceSnapshot:
        if hasattr(self, '_snap'): return self._snap
        # モニターが無効な場合は都度取得
        return self._capture_snapshot()

    def _loop(self):
        while not self._stop.is_set():
            self._snap = self._capture_snapshot()
            # 指定された間隔で待機
            self._stop.wait(timeout=self.sample_sec)

class BeamGovernor:
    # リソース状況と残り時間に基づいてビーム幅を動的に調整
    # [最適化] ARC 2025 12Hルール向けにパラメータを調整（積極性と安全性のバランス）
    def __init__(self,
                 # RAM目標使用率: 70% -> 75% (より積極的に使用)
                 target_ram_frac: float = 0.75,
                 # RAMハードリミット: 85% -> 88% (緊急回避ライン)
                 hard_ram_frac: float = 0.88,
                 # VRAM設定（GPU利用時）
                 target_vram_frac: float = 0.80, # 0.75 -> 0.80
                 hard_vram_frac: float = 0.92,   # 0.90 -> 0.92
                 # ビーム幅の範囲
                 min_beam: int = 4,
                 max_beam: int = 128, # デフォルト上限を64から引き上げ
                 min_keep: int = 16,
                 max_keep: int = 2048, # デフォルト上限を1024から引き上げ
                 # 時間経過による減衰開始点: 70% -> 80% (探索時間を長く確保)
                 decay_after_ratio: float = 0.80):

        self.target_ram = target_ram_frac; self.hard_ram = hard_ram_frac
        self.target_vram = target_vram_frac; self.hard_vram = hard_vram_frac
        self.min_beam = min_beam; self.max_beam = max_beam
        self.min_keep = min_keep; self.max_keep = max_keep
        self.decay_after = decay_after_ratio

    def suggest(self, snap: ResourceSnapshot, default_beam:int, default_keep:int, elapsed:float, time_budget:float) -> Tuple[int,int,str]:
        # 入力値のサニタイズ（デフォルト値を使用し、最大/最小範囲でクリップ）
        beam = max(self.min_beam, min(self.max_beam, int(default_beam)))
        keep = max(self.min_keep, min(self.max_keep, int(default_keep)))
        reason = "ok"

        # リソース使用率の取得
        ramf = snap.ram_frac(); vramfs = snap.vram_fracs()
        has_gpu = bool(vramfs); vramf = max(vramfs) if has_gpu else 0.0

        # --- メモリ管理フェーズ ---
        # 制御ロジックを整理し、ハードリミットを最優先で処理

        # 1. ハードリミット（危険域）に基づく強制削減
        if ramf >= self.hard_ram:
            # RAMが限界に近い場合、ビーム幅を大幅に削減（例: 1/3）
            reduction_factor = 0.33
            beam = int(math.floor(beam * reduction_factor))
            keep = int(math.floor(keep * reduction_factor))
            reason = f"hard_ram({ramf:.2f})"
        elif has_gpu and vramf >= self.hard_vram:
             # VRAMが限界に近い場合（例: 1/2）
            reduction_factor = 0.5
            beam = int(math.floor(beam * reduction_factor))
            keep = int(math.floor(keep * reduction_factor))
            reason = f"hard_vram({vramf:.2f})"

        # 2. ソフトリミット（目標値）に基づく調整
        else:
            # [最適化] 目標値を超えた場合、ハードリミットまでの距離に応じて線形に削減係数を計算
            # この方式は、逆比例よりも高負荷時の制御が安定する。
            # 例: target=0.75, hard=0.88。ramf=0.80の場合、(0.88-0.80)/(0.88-0.75) = 0.08/0.13 ≈ 0.61
            ram_factor = 1.0
            if ramf > self.target_ram:
                # 最小係数を0.5（半減）として保証
                ram_factor = max(0.5, (self.hard_ram - ramf) / (self.hard_ram - self.target_ram))

            vram_factor = 1.0
            if has_gpu and vramf > self.target_vram:
                vram_factor = max(0.5, (self.hard_vram - vramf) / (self.hard_vram - self.target_vram))

            soft_factor = min(ram_factor, vram_factor)

            if soft_factor < 1.0:
                beam = int(math.floor(beam * soft_factor))
                keep = int(math.floor(keep * soft_factor))
                reason = f"soft({soft_factor:.2f})"

        # --- 時間管理フェーズ ---

        # 3. 時間経過に基づく調整（探索終盤）
        if time_budget > 0 and time_budget < float('inf'):
            ratio = elapsed / time_budget
            if ratio >= self.decay_after:
                dec_range = (1.0 - self.decay_after)
                # 減衰率（最小0.25まで許容）。終盤は探索を収束させる。
                dec = max(0.25, 1.0 - (ratio - self.decay_after) / dec_range) if dec_range > 1e-6 else 0.25
                beam = int(math.floor(beam * dec))
                keep = int(math.floor(keep * dec))
                if reason == "ok": reason = f"late({dec:.2f})"
                else: reason += f"|late({dec:.2f})"

        # 最終的な範囲調整（最小値保証）
        beam = max(self.min_beam, min(self.max_beam, beam))
        keep = max(self.min_keep, min(self.max_keep, keep))

        return beam, keep, reason

# ==============================================================================
# 1. kutos/config.py
# ==============================================================================
# (OMUX004o: 探索パラメータ強化・動的調整)
class KUTOSConfig:
    # メインプロセスから注入された設定値を使用
    # KUT30のビーム幅
    KUT30_BEAM_K = CFG_INJECTED["MAX_BEAMS"]

    # AutoTuneで制御されるその他のパラメータ
    AUTOTUNE_CFG = CFG_INJECTED

    # 静的パラメータ
    # [最適化] 12時間ルール対応: GEOM_MATCH_MAX_PAIRS をGPU/CPUと時間予算に応じて調整 (精度向上)
    # 延長されたタスク時間予算を活用し、探索密度を向上させる。
    # 元の200から、CPUでは300、GPUでは800へ引き上げ。
    _is_gpu = GPU_ENABLED # ワーカー内でのGPU判定フラグを利用
    GEOM_MATCH_MAX_PAIRS = 800 if _is_gpu else 300

    GEOM_MATCH_IOU_THRESHOLD = 0.95

CONFIG = KUTOSConfig()

# ==============================================================================
# 2. kutos/common.py
# ==============================================================================
# (OMUX004o: メモリ効率改善、データ整合性・安全性強化)

try:
    Grid = np.ndarray
except AttributeError:
    Grid = Any

TaskPredictionsInternal = List[Dict[str, Grid]]

# ARCグリッドの最大サイズ（提出ルール準拠）
MAX_GRID_H, MAX_GRID_W = 30, 30

def grid_to_numpy(grid: Any) -> Grid:
    # リストをNumPy配列（またはDummyNp）に変換
    # [最適化] メモリ効率改善: dtypeをuint8に変更。ARCグリッド(0-9)に最適化し、メモリ消費を大幅削減。
    if isinstance(np, DummyNp): return grid if isinstance(grid, list) else [[0]]

    try:
        # uint8の存在確認（安全対策）
        if not hasattr(np, 'uint8'):
             return _grid_to_numpy_fallback(grid, int)

        if grid is None:
            return np.zeros((1, 1), dtype=np.uint8)

        if isinstance(grid, np.ndarray):
            # 既存のndarrayの場合、uint8に変換して返す（copy=Falseで効率化）
            return grid.astype(np.uint8, copy=False)

        # リストからの変換時もuint8を指定
        np_grid = np.array(grid, dtype=np.uint8)

        # 空の入力や不正な形状のハンドリング
        if np_grid.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # [最適化] 安全性強化: 形状を2Dに強制
        if np_grid.ndim != 2:
            if np_grid.ndim == 0:
                # スカラー値
                np_grid = np.array([[np_grid.item()]], dtype=np.uint8)
            elif np_grid.ndim == 1:
                # 1D配列（列ベクトルとして扱う）
                np_grid = np_grid.reshape((-1, 1))
            else:
                # 3次元以上は安全にリシェイプを試みる
                try:
                    np_grid = np_grid.reshape(np_grid.shape[0], -1)
                except ValueError:
                    return np.zeros((1, 1), dtype=np.uint8)

        return np_grid
    except Exception:
        # 数値以外のデータが含まれる場合や変換失敗時
        if hasattr(np, 'uint8'):
            return np.zeros((1, 1), dtype=np.uint8)
        return _grid_to_numpy_fallback(grid, int)

def _grid_to_numpy_fallback(grid: Any, dtype) -> Grid:
    # uint8が使えない環境用のフォールバック
    try:
        if isinstance(grid, np.ndarray): return grid.astype(dtype, copy=False)
        return np.array(grid if grid is not None else [[0]], dtype=dtype)
    except:
        return np.array([[0]], dtype=dtype)


def numpy_to_list(array: Grid) -> List[List[int]]:
    # NumPy配列をリストに変換（提出用に厳密にサニタイズ）
    # [最適化] 長時間実行の安全性確保: 形状(30x30)と値(0-9整数)を保証する

    if array is None: return [[0]]

    # NumPyが利用可能な場合の処理（推奨パス）
    if not isinstance(np, DummyNp):
        try:
            if not isinstance(array, np.ndarray):
                np_array = np.array(array)
            else:
                np_array = array

            # 1. 形状のサニタイズ（次元とサイズ）
            if np_array.ndim != 2:
                # 2次元以外は整形を試みる
                if np_array.size == 0: return [[0]]
                if np_array.ndim == 0:
                    np_array = np.array([[np_array.item()]])
                elif np_array.ndim == 1:
                     # 1D配列は行ベクトルとして扱う
                    np_array = np_array.reshape((1, -1))
                else:
                    # 3D以上は扱わない（安全なフォールバック）
                    return [[0]]

            H, W = np_array.shape

            # 30x30以内にクリッピング
            if H > MAX_GRID_H or W > MAX_GRID_W:
                H_clip, W_clip = min(H, MAX_GRID_H), min(W, MAX_GRID_W)
                np_array = np_array[:H_clip, :W_clip]
                H, W = H_clip, W_clip

            # 最小サイズ保証 (1x1)
            if H == 0 or W == 0: return [[0]]

            # 2. 値のサニタイズ (整数化と0-9の範囲へのクランプ)
            # 数値型であることを確認
            if not np.issubdtype(np_array.dtype, np.number):
                return [[0]]

            # 整数化（浮動小数点数の場合）
            if not np.issubdtype(np_array.dtype, np.integer):
                np_array = np_array.astype(int)

            # 0-9の範囲にクランプ
            np_array = np.clip(np_array, 0, 9)

            # .tolist()でPythonネイティブのintへ変換
            return np_array.tolist()

        except Exception:
            # NumPy処理中のエラー発生時
            return [[0]]

    # NumPyが利用不可（DummyNp）または入力がリストの場合の処理（フォールバック）
    if isinstance(array, list):
        try:
            # 形状のサニタイズ（簡易版）
            if not array: return [[0]]

            # 2次元リストか確認
            if not isinstance(array[0], list):
                 # 1Dリストの場合は2Dにラップ
                 array = [array]

            H = len(array)
            # 最初の行で幅を推定
            W = len(array[0]) if H > 0 and isinstance(array[0], list) else 0
            if H == 0 or W == 0: return [[0]]

            H_clip, W_clip = min(H, MAX_GRID_H), min(W, MAX_GRID_W)
            sanitized = []
            for r in range(H_clip):
                row = array[r]
                s_row = []
                # 行がリストでない場合はスキップ
                if not isinstance(row, list): continue

                # W_clipと実際の行の長さの小さい方を使用
                for c in range(min(W_clip, len(row))):
                    val = row[c]
                    # 値のサニタイズ（簡易版）
                    try:
                        ival = int(val)
                        s_row.append(max(0, min(9, ival)))
                    except:
                        s_row.append(0)
                # 行が短すぎる場合のパディング
                while len(s_row) < W_clip: s_row.append(0)
                sanitized.append(s_row)

            if not sanitized: return [[0]]
            return sanitized
        except Exception:
            return [[0]]

    return [[0]]

class ARCTask:
    __slots__ = ('task_id', 'train_pairs', 'test_inputs')
    def __init__(self, task_data: Dict, task_id: str):
        self.task_id = task_id
        # パース時にgrid_to_numpy（uint8化と安全確保）が適用される
        self.train_pairs = self._parse_train_pairs(task_data.get("train", []))
        self.test_inputs: List[Grid] = self._parse_test_inputs(task_data.get("test", []))

    def _parse_train_pairs(self, pairs_data: List[Dict]) -> List[Dict]:
        pairs = []
        for p in pairs_data:
            if isinstance(p, dict) and "input" in p and "output" in p:
                try:
                    # grid_to_numpyで安全に変換
                    inp = grid_to_numpy(p["input"])
                    out = grid_to_numpy(p["output"])
                    pairs.append({"input": inp, "output": out})
                except Exception: continue
        return pairs

    def _parse_test_inputs(self, test_data: List[Dict]) -> List[Grid]:
        inputs = []
        try:
            for item in test_data:
                input_data = item.get("input")
                # grid_to_numpyはNoneや空入力を安全に処理する
                inputs.append(grid_to_numpy(input_data))
        except Exception:
             # パース失敗時の安全なフォールバック
             if not inputs: inputs.append(grid_to_numpy([[0]]))
        return inputs

class CandidateSolution:
    __slots__ = ('source', 'confidence', 'solver_logic')
    def __init__(self, source: str, confidence: float, solver_logic: Callable[[Grid], Grid]):
        self.source, self.confidence, self.solver_logic = source, confidence, solver_logic

    def solve(self, input_grid: Grid) -> Optional[Grid]:
        try:
            # ソルバーロジックを実行
            result = self.solver_logic(input_grid)

            # [最適化] ロバストネス強化: 結果が有効なグリッド形式か厳密に確認
            if isinstance(np, DummyNp):
                # DummyNp環境: 2Dリストであることを確認
                if isinstance(result, list) and result and isinstance(result[0], list):
                    return result
                return None

            # NumPy環境: ndarray形式を期待
            if isinstance(result, np.ndarray):
                # 2次元配列であり、空でなく、数値型であることを確認
                if result.ndim == 2 and result.size > 0 and np.issubdtype(result.dtype, np.number):
                    return result

            # リスト形式で返された場合も許容し、安全に変換を試みる
            if isinstance(result, list):
                try:
                    converted = grid_to_numpy(result)
                    if converted.ndim == 2 and converted.size > 0:
                        return converted
                except: pass

            return None
        except Exception:
            # 実行時エラーはキャッチし、Noneを返す（安定性重視）
            return None

# ==============================================================================
# 3. kutos/geometry_utils.py
#    幾何学的な計算とオブジェクト操作のためのユーティリティ
# ==============================================================================
# (OMUX004o: ハッシュ計算の安定化・効率化、サンプリング効率改善、実装修正)

def calculate_holes(mask: Grid) -> int:
    # SciPyのbinary_fill_holesを使用して穴の数を計算
    if scipy_binary_fill_holes is None or isinstance(np, DummyNp): return 0

    # 入力データの安全確認
    if not hasattr(mask, 'shape') or mask.ndim != 2 or (hasattr(mask, 'size') and mask.size == 0): return 0

    try:
        # maskがbooleanでない場合は変換（安全性を確保）
        if hasattr(mask, 'astype') and (not hasattr(mask, 'dtype') or mask.dtype != bool):
             mask_bool = mask.astype(bool)
        else:
             mask_bool = mask

        filled = scipy_binary_fill_holes(mask_bool)
        holes_mask = filled & (~mask_bool)

        if scipy_label:
            # [最適化] 構造要素を指定して8連結性を明示（安定性向上）
            structure = np.ones((3, 3), dtype=int) if hasattr(np, 'ones') else None
            _, num_holes = scipy_label(holes_mask, structure=structure)
            return int(num_holes)
        return 0
    except Exception:
        return 0

# D4対称変換（回転・反転）
D4_TRANSFORMS = {
    'I': lambda m: m, 'R90': lambda m: np.rot90(m, k=1), 'R180': lambda m: np.rot90(m, k=2), 'R270': lambda m: np.rot90(m, k=3),
    'FH': lambda m: np.fliplr(m), 'FV': lambda m: np.flipud(m), 'FD': lambda m: np.transpose(m),
    # [修正] 反対角線反転(FAD)の実装を修正。np.flipud(np.transpose(m))は誤り。
    'FAD': lambda m: np.rot90(np.fliplr(m), k=1)
}

def _mask_to_bytes(mask: Grid) -> bytes:
    # マスクデータを一意なバイト列に変換（ハッシュ計算用）
    # [最適化] 効率と安定性の向上：マスクをブール値(0/1)として扱い、uint8でバイト列化する
    # これにより、マスク内の値（色）によらず形状のみに基づくハッシュが可能になる。
    if isinstance(np, DummyNp) or (hasattr(mask, 'size') and mask.size == 0):
        return str(mask).encode()

    try:
        # 1. マスクをブール値(0/1)のuint8に変換。
        if mask.dtype == np.bool_:
             mask_u8 = mask.astype(np.uint8, copy=False)
        # uint8だが1より大きい値が含まれる場合、または他の型の場合
        elif mask.dtype != np.uint8 or np.max(mask) > 1:
             mask_u8 = (mask > 0).astype(np.uint8, copy=False)
        else:
             mask_u8 = mask

        # 2. メモリレイアウトをC連続に統一（一貫性確保）
        if hasattr(mask_u8, 'flags') and not mask_u8.flags['C_CONTIGUOUS']:
            mask_u8 = np.ascontiguousarray(mask_u8)

        # 3. ヘッダー（形状情報）とバイト列を結合
        header = f"{mask_u8.shape[0]}x{mask_u8.shape[1]}:".encode()
        return header + mask_u8.tobytes()

    except Exception:
        # フォールバック（低速だが安全）
        try:
            header = f"fallback{mask.shape[0]}x{mask.shape[1]}:".encode()
            return header + bytes(mask.tolist())
        except:
            return str(mask).encode()

def calculate_signature(mask: Grid, d4_invariant: bool = False) -> Tuple[str, str]:
    # マスクの幾何学的シグネチャ（ハッシュ）を計算
    if isinstance(np, DummyNp) or (hasattr(mask, 'size') and mask.size == 0): return str(hash(str(mask))), 'I'

    try:
        if not d4_invariant:
            # D4非不変（向きを区別）
            b = _mask_to_bytes(mask)
            return hashlib.sha1(b).hexdigest(), 'I'

        # D4不変（回転・反転しても同じとみなす）
        # 辞書順で最小となるバイト列表現を採用
        best_bytes, best_name = None, None
        for name, transform_fn in D4_TRANSFORMS.items():
            try:
                transformed_mask = transform_fn(mask)
                bb = _mask_to_bytes(transformed_mask)
            except Exception: continue

            if best_bytes is None or bb < best_bytes:
                best_bytes, best_name = bb, name

        if best_bytes:
            return hashlib.sha1(best_bytes).hexdigest(), best_name or 'I'
        else:
            return "error_hash", 'I'
    except Exception:
        return "error_hash", 'I'

def _sample_points(points: np.ndarray, k: int) -> np.ndarray:
    # 点群からk個の点をランダムサンプリング（GeomMatch用）
    # [最適化] 効率改善: np.random.choiceを使用
    if isinstance(np, DummyNp) or not hasattr(points, 'shape') or points.ndim != 2: return points
    N = points.shape[0]

    # サンプル数が点数以下であることを保証
    k_safe = min(k, N)
    if k_safe <= 0:
        # 空の配列を安全に返す
        return np.empty((0, points.shape[1]), dtype=points.dtype) if hasattr(np, 'empty') else points

    if N == k_safe: return points

    try:
        # replace=Falseで重複なしサンプリング。NumPyネイティブで高速。
        indices = np.random.choice(N, k_safe, replace=False)
        return points[indices]
    except Exception:
        # フォールバック（Python標準のrandom.sample）
        try:
            indices = random.sample(range(N), k_safe)
            return points[indices]
        except:
            # 最終フォールバック（先頭K個）
            return points[:k_safe]

def align_components(A_obj, B_obj, max_pairs: int = CONFIG.GEOM_MATCH_MAX_PAIRS) -> Dict[str, object]:
    # 2つのオブジェクト間の最適な幾何学的位置合わせ（D4変換＋平行移動）を探索
    # [最適化] 安定性向上: 入力オブジェクトの堅牢なチェック
    if isinstance(np, DummyNp): return {'transform': None, 'dx': 0, 'dy': 0, 'iou': 0.0}

    # 入力オブジェクトの属性チェック
    if not hasattr(A_obj, 'pixels') or not hasattr(B_obj, 'pixels') or not hasattr(A_obj, 'mask'):
         return {'transform': None, 'dx': 0, 'dy': 0, 'iou': 0.0}

    A_pixels_abs = A_obj.pixels
    B_pixels_abs = B_obj.pixels
    A_mask = A_obj.mask

    # 属性が期待通りの形式（ndarray）であり、空でないことを確認
    if not isinstance(A_pixels_abs, np.ndarray) or not isinstance(B_pixels_abs, np.ndarray) or not isinstance(A_mask, np.ndarray) or \
       A_pixels_abs.size == 0 or B_pixels_abs.size == 0 or A_mask.size == 0:
        return {'transform': None, 'dx': 0, 'dy': 0, 'iou': 0.0}

    # Bの座標セット（高速な存在確認用）
    try:
        Bset_global = set(map(tuple, B_pixels_abs))
    except TypeError:
         return {'transform': None, 'dx': 0, 'dy': 0, 'iou': 0.0}

    lenB = len(Bset_global)
    lenA = A_pixels_abs.shape[0]

    # 探索空間の削減：アンカーポイントのサンプリング数kを決定 (k*k ≈ max_pairs)
    k = max(1, int(math.sqrt(max_pairs)))

    best = {'transform': None, 'dx': 0, 'dy': 0, 'iou': -1.0}

    # Bのアンカーポイントをサンプリング
    B_anchors_global = _sample_points(B_pixels_abs, k)

    # 全てのD4変換について探索
    for tname, transform_fn in D4_TRANSFORMS.items():
        try:
            # Aのマスクを変換し、変換後のローカル座標を取得
            A_mask_T = transform_fn(A_mask)
            A_pixels_T_local = np.argwhere(A_mask_T)

            # 変換後のAのアンカーポイントをサンプリング
            A_anchors_relative = _sample_points(A_pixels_T_local, k)

            if A_anchors_relative.size == 0 or B_anchors_global.size == 0: continue

            # アンカーペア間の差分ベクトル（平行移動量）を計算
            # NumPyブロードキャストを利用して効率的に計算
            deltas = B_anchors_global[:, None, :] - A_anchors_relative[None, :, :]
            # 重複する差分ベクトルを除去
            unique_deltas = np.unique(deltas.reshape(-1, 2), axis=0)

            # 各平行移動量についてIoUを計算
            for dy, dx in unique_deltas:
                # Aを平行移動
                A_translated = A_pixels_T_local + [dy, dx]

                try:
                    A_set_final = set(map(tuple, A_translated))
                except TypeError: continue

                # IoU計算
                intersection = len(A_set_final.intersection(Bset_global))
                union = lenA + lenB - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou > best['iou']:
                    best = {'transform': tname, 'dx': int(dx), 'dy': int(dy), 'iou': iou}

                # 完全一致したら探索終了
                if iou >= 0.9999:
                    best['iou'] = 1.0
                    return best
        except Exception:
             # 変換や計算中のエラーは無視して続行
            continue
    return best
# ==============================================================================
# 4. kutos/core_modules.py
#    シーン表現、抽象化、一貫性チェックなどのコアモジュール
# ==============================================================================
# (OMUX004o: メモリ効率改善(int16座標)、背景色検出、抽象化堅牢化、一貫性評価改善)

class SceneObject:
    # シーン内の単一オブジェクト表現
    # [最適化] area（面積）属性を追加
    __slots__ = ('mask', 'color', 'bbox', 'pixels', 'holes', 'sig', 'sig_d4', 'area')

    def __init__(self, mask: np.ndarray, color: int, bbox: Tuple[slice, slice]):
        self.mask, self.color, self.bbox = mask, color, bbox
        self.area = 0

        # [最適化] メモリ効率改善: ピクセル座標(pixels)の計算とデータ型指定(int16)
        if isinstance(np, DummyNp):
             self.pixels = []; self._initialize_metadata_fallback(); return

        # 座標データ型を設定。30x30なのでint16で十分安全かつ効率的。
        COORD_DTYPE = np.int16 if hasattr(np, 'int16') else int

        try:
            if isinstance(mask, np.ndarray) and mask.size > 0 and mask.any():
                # bboxの開始位置を安全に取得 (Noneの場合は0)
                start_r = bbox[0].start or 0
                start_c = bbox[1].start or 0

                local_coords = np.argwhere(mask)
                self.area = local_coords.shape[0]

                # ベクトル化された計算でグローバル座標を算出
                offset = np.array([start_r, start_c], dtype=COORD_DTYPE)
                # 安全にキャスト
                self.pixels = (local_coords + offset).astype(COORD_DTYPE, copy=False)
            else:
                self.pixels = np.empty((0, 2), dtype=COORD_DTYPE)
        except Exception:
             # 計算エラーやキャスト失敗時のフォールバック
             self.pixels = np.empty((0, 2), dtype=COORD_DTYPE)

        self._calculate_metadata()

    def _initialize_metadata_fallback(self):
        self.holes = 0; self.sig = "undefined"; self.sig_d4 = "undefined"

    def _calculate_metadata(self):
        # メタデータ（穴の数、シグネチャ）の計算
        self._initialize_metadata_fallback()

        if isinstance(np, DummyNp) or not isinstance(self.mask, np.ndarray) or self.area == 0: return

        try:
            # geometry_utilsの関数を利用
            self.holes = calculate_holes(self.mask)
            self.sig, _ = calculate_signature(self.mask, d4_invariant=False)
            self.sig_d4, _ = calculate_signature(self.mask, d4_invariant=True)
        except Exception:
            pass # 計算失敗時はデフォルト値を維持

class SceneGraph:
    # グリッド全体のシーン表現（オブジェクトの集合）
    def __init__(self, grid: Grid):
        self.objects: List[SceneObject] = []
        self.background_color: int = 0

        # 入力グリッドの初期処理と検証（grid_to_numpyで安全に正規化・uint8化）
        normalized_grid = grid_to_numpy(grid)

        if normalized_grid is None or (hasattr(normalized_grid, 'size') and normalized_grid.size == 0):
            self.grid = grid_to_numpy([[0]]); self.height, self.width = 1, 1; return

        self.grid = normalized_grid

        if isinstance(np, DummyNp):
            self.height = len(self.grid) if self.grid else 0
            self.width = len(self.grid[0]) if self.height > 0 and isinstance(self.grid[0], list) else 0
            return

        # NumPy環境での初期化
        # grid_to_numpyで2D保証済み
        self.height, self.width = self.grid.shape

        # [最適化] 背景色の動的検出（最頻値）
        try:
            # bincountで高速に最頻値を計算（0-9の範囲に最適化されたuint8グリッドを利用）
            if self.grid.dtype == np.uint8 and self.grid.max() < 256:
                 counts = np.bincount(self.grid.ravel())
                 self.background_color = int(np.argmax(counts))
            else:
                # フォールバック（通常パス）
                colors, counts = np.unique(self.grid, return_counts=True)
                self.background_color = int(colors[np.argmax(counts)])
        except Exception:
            self.background_color = 0

        # オブジェクト抽出
        self.objects = self._extract_objects(self.grid)

    def _extract_objects(self, grid: Grid) -> List[SceneObject]:
        # 連結成分ラベリングによるオブジェクト抽出
        # [最適化] 背景色対応とロバストネス強化（フォールバック導入）

        # 1. SciPyによる抽出試行
        if scipy_label is not None and scipy_find_objects is not None:
            try:
                structure = np.ones((3, 3), dtype=int)
                # 背景色以外の領域をマスク化
                foreground_mask = (grid != self.background_color)

                labeled, n_labels = scipy_label(foreground_mask, structure=structure)
                if n_labels == 0: return []

                slices = scipy_find_objects(labeled)

                objs = []
                for i in range(n_labels):
                    s = slices[i]
                    if s is None: continue

                    # マスク抽出
                    mask = (labeled[s] == i + 1)

                    # 主要色の決定
                    try:
                        object_pixels = grid[s][mask]
                        if object_pixels.size == 0: continue

                        # 最頻値を計算
                        colors, counts = np.unique(object_pixels, return_counts=True)
                        main_color = int(colors[np.argmax(counts)])
                        objs.append(SceneObject(mask, main_color, s))
                    except Exception:
                        continue
                return objs

            except Exception:
                # SciPy実行時エラー発生時はフォールバックへ
                pass

        # 2. フォールバック（SciPy不在または失敗時）
        # フォアグラウンド全体を単一のオブジェクトとして扱う
        try:
            mask = (grid != self.background_color)
            if mask.any():
                colors, counts = np.unique(grid[mask], return_counts=True)
                if len(colors) > 0:
                    main_color = int(colors[np.argmax(counts)])
                    # Bboxはグリッド全体
                    bbox = (slice(0, self.height), slice(0, self.width))
                    return [SceneObject(mask, main_color, bbox)]
        except Exception:
            pass

        return []

    def to_grid(self) -> Grid:
        # シーングラフからグリッドを再構築
        if isinstance(np, DummyNp): return self.grid

        # [最適化] メモリ効率維持: 再構築時もuint8を使用し、背景色で初期化
        dtype = np.uint8 if hasattr(np, 'uint8') else int
        new_grid = np.full((self.height, self.width), self.background_color, dtype=dtype)

        for obj in self.objects:
            # ピクセルデータが有効か確認 (SceneObjectでint16化済み)
            if not hasattr(obj.pixels, 'shape') or obj.pixels.shape[0] == 0: continue

            try:
                coords = obj.pixels
                # 境界チェック（int16なので負の値も考慮）
                valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < self.height) & \
                             (coords[:, 1] >= 0) & (coords[:, 1] < self.width)

                if np.any(valid_mask):
                    valid_coords = coords[valid_mask]
                    # 一括で色を設定
                    new_grid[valid_coords[:, 0], valid_coords[:, 1]] = obj.color
            except Exception:
                # 再構築中のエラーは無視
                continue
        return new_grid

class AbstractionModule:
    # グリッドからシーングラフへの抽象化
    def abstract(self, grid: Grid) -> SceneGraph:
        # SceneGraphの初期化で抽象化処理が実行される
        return SceneGraph(grid)

class AugmentationFramework:
    # データ拡張（D4対称性）
    def get_d4_symmetries(self) -> List[Tuple[str, Callable, Callable]]:
        symmetries = []

        if isinstance(np, DummyNp):
             # NumPyが使えない場合は恒等変換のみ
             return [("I", lambda g: g, lambda g: g)]

        # 4方向の回転と水平反転の組み合わせ (8通り)
        for rot in range(4):
            for flip in [False, True]:
                # クロージャを利用して変換関数を生成
                def fwd(g, r=rot, f=flip):
                    # 順方向変換
                    try:
                        transformed = np.rot90(g, k=r)
                        if f: transformed = np.fliplr(transformed)
                        return transformed
                    except: return g # 失敗時は入力を返す

                def inv(g, r=rot, f=flip):
                    # 逆方向変換
                    try:
                        transformed = g
                        if f: transformed = np.fliplr(transformed)
                        # 逆回転
                        transformed = np.rot90(transformed, k=(4-r)%4)
                        return transformed
                    except: return g

                symmetries.append((f"R{rot*90}{'F' if flip else ''}", fwd, inv))
        return symmetries

class ConsistencyChecker:
    # 解候補の一貫性評価
    def __init__(self, aug_fw: AugmentationFramework):
        self.symmetries = aug_fw.get_d4_symmetries()
        self.n_symmetries = len(self.symmetries)

    def evaluate(self, cand: CandidateSolution, test_input: Grid) -> float:
        # D4対称性を用いた一貫性スコアの計算
        # [最適化] 評価指標の改善: 乗算スコアから平均一貫性スコア(0.0-1.0)へ変更

        base_out = cand.solve(test_input)
        if base_out is None: return 0.0

        # NumPyが使えない環境のフォールバック
        if isinstance(np, DummyNp):
             return 0.5 # 評価不能なため中間値

        consistent_count = 0

        # 全ての対称変換についてテスト
        for _, fwd, inv in self.symmetries:
            try:
                # 入力を変換(fwd) -> 解を適用(solve) -> 逆変換(inv)
                aug_in = fwd(test_input)
                aug_out = cand.solve(aug_in)

                if aug_out is None:
                    continue # 解が得られなかった場合は不一致

                rev_out = inv(aug_out)

                # 結果比較 (元の出力と一致するか)
                # 形状チェック
                if not hasattr(rev_out, 'shape') or rev_out.shape != base_out.shape:
                    continue

                # 内容チェック
                if np.array_equal(base_out, rev_out):
                    consistent_count += 1

            except Exception:
                # 評価中のエラーは不一致とみなす
                continue

        # 平均一貫性スコアを返す
        return float(consistent_count) / self.n_symmetries if self.n_symmetries > 0 else 0.0

# ==============================================================================
# 5. kutos/engines.py (Induction, Transduction)
#    帰納的プログラム合成と演繹的ヒューリスティクスエンジン
# ==============================================================================
# (OMUX004o: 帰納探索能力向上(IoUマッチング追加)、演繹エンジン精度向上・堅牢化)

# --- Engine 1: Induction (Program Synthesis) ----------------------------------
class ObjectCentricDSL:
    # 簡単なDSL実行エンジン
    # [最適化] 拡張されたDSL（シグネチャベースのルール適用）に対応
    def execute(self, prog: List[Dict], scene: SceneGraph) -> SceneGraph:
        # SceneGraphはイミュータブルとして扱い、コピーして変更する
        # deepcopyは遅いため、必要な部分のみコピーする
        new_scene = copy.copy(scene)
        # SceneObjectもコピーする（color属性を変更するため）
        new_scene.objects = [copy.copy(obj) for obj in scene.objects]

        for step in prog:
            op = step.get("op")
            if op == "recolor":
                # 色の変更操作
                target_color = step.get("target_color")
                new_color = step.get("new_color")
                # マッチング基準の取得
                match_sig = step.get("match_sig")
                match_sig_d4 = step.get("match_sig_d4")

                if target_color is not None and new_color is not None:
                    try:
                        t_c, n_c = int(target_color), int(new_color)
                    except: continue

                    for obj in new_scene.objects:
                        # 色が一致するか確認
                        is_target = (obj.color == t_c)

                        # シグネチャ条件を確認
                        matches_criteria = True
                        if match_sig is not None:
                            matches_criteria = (obj.sig == match_sig)
                        elif match_sig_d4 is not None:
                            matches_criteria = (obj.sig_d4 == match_sig_d4)
                        # どちらもNoneの場合は色のみで判定 (IoUベースマッチング時)

                        if is_target and matches_criteria:
                            obj.color = n_c
        return new_scene

class KUTInductionEngine:
    # オブジェクト間のプロパティ変化から一貫したルール（DSLプログラム）を帰納
    def __init__(self, am: AbstractionModule):
        self.abstraction, self.dsl = am, ObjectCentricDSL()
        # [最適化] IoUマッチングの閾値
        self.IOU_MATCH_THRESHOLD = 0.85

    def _calculate_iou(self, objA: SceneObject, objB: SceneObject) -> float:
        # 2つのオブジェクト間のIoU（Intersection over Union）を計算
        if isinstance(np, DummyNp) or objA.area == 0 or objB.area == 0: return 0.0

        try:
            # 座標セットを作成 (pixelsはint16)
            setA = set(map(tuple, objA.pixels))
            setB = set(map(tuple, objB.pixels))

            intersection = len(setA.intersection(setB))
            # SceneObject.areaを利用
            union = objA.area + objB.area - intersection

            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def _match_objects(self, in_s: SceneGraph, out_s: SceneGraph) -> Tuple[Dict[int, int], str]:
        # 入出力のオブジェクト間のマッチング
        # [最適化] 3段階マッチング戦略: 1. Exact(sig) -> 2. D4 Invariant(sig_d4) -> 3. High IoU
        in_o, out_o = in_s.objects, out_s.objects

        # --- Stage 1: Exact Match (sig) ---
        matches, used_out = {}, set()
        match_type = "exact"

        out_h_sig = defaultdict(list)
        for i, o in enumerate(out_o):
            out_h_sig[o.sig].append(i)

        for i, in_obj in enumerate(in_o):
            if in_obj.sig in out_h_sig:
                for cand_idx in out_h_sig[in_obj.sig]:
                    if cand_idx not in used_out:
                        matches[i] = cand_idx
                        used_out.add(cand_idx)
                        break

        # 完全マッチングできたか確認
        if len(matches) == len(in_o) and len(matches) == len(out_o):
            return matches, match_type

        # --- Stage 2: D4 Invariant Match (sig_d4) ---
        # AutoTuneでInductionが有効な場合のみ試行（計算コスト考慮）
        if CONFIG.AUTOTUNE_CFG.get("ENABLE_INDUCTION", False):
            matches, used_out = {}, set()
            match_type = "d4_invariant"

            out_h_sig_d4 = defaultdict(list)
            for i, o in enumerate(out_o):
                out_h_sig_d4[o.sig_d4].append(i)

            for i, in_obj in enumerate(in_o):
                if in_obj.sig_d4 in out_h_sig_d4:
                    for cand_idx in out_h_sig_d4[in_obj.sig_d4]:
                        if cand_idx not in used_out:
                            matches[i] = cand_idx
                            used_out.add(cand_idx)
                            break

            # 完全マッチングできたか確認
            if len(matches) == len(in_o) and len(matches) == len(out_o):
                return matches, match_type

        # --- Stage 3: High IoU Match (Greedy) ---
        matches, used_out = {}, set()
        match_type = "iou_based"

        iou_matrix = []
        for i, in_obj in enumerate(in_o):
            for j, out_obj in enumerate(out_o):
                iou = self._calculate_iou(in_obj, out_obj)
                if iou >= self.IOU_MATCH_THRESHOLD:
                    iou_matrix.append((iou, i, j))

        # IoUが高い順に貪欲にマッチング
        iou_matrix.sort(key=lambda x: x[0], reverse=True)
        for iou, i, j in iou_matrix:
            if i not in matches and j not in used_out:
                matches[i] = j
                used_out.add(j)

        # IoUベースでは完全マッチングを必須としない
        return matches, match_type


    def _synthesize_program(self, task: ARCTask) -> Tuple[List[Dict], float]:
        # 全ての学習ペアで一貫した色の変化ルールを帰納
        # [最適化] 緩和されたマッチング基準に対応し、信頼度を計算

        rules = {} # キー: (色, シグネチャタイプ, シグネチャ値), 値: 新しい色
        final_match_type = None
        ok = True

        for pair_idx, pair in enumerate(task.train_pairs):
            try:
                in_s = self.abstraction.abstract(pair['input'])
                out_s = self.abstraction.abstract(pair['output'])
                matches, match_type = self._match_objects(in_s, out_s)

                # マッチングタイプの一貫性をチェック
                if final_match_type is None:
                    final_match_type = match_type
                elif final_match_type != match_type:
                    # ペア間でマッチング基準が異なるとルールが一貫しない
                    ok = False; break

                # マッチングが空で、かつどちらかにオブジェクトが存在する場合
                if not matches and (in_s.objects or out_s.objects):
                    ok = False; break

                for i, o_idx in matches.items():
                    # インデックスの安全チェック
                    if i >= len(in_s.objects) or o_idx >= len(out_s.objects): ok = False; break

                    in_obj = in_s.objects[i]
                    out_obj = out_s.objects[o_idx]

                    if in_obj.color != out_obj.color:
                        # ルールのキーを決定（マッチング基準に応じて精度を変える）
                        if match_type == "exact":
                            rule_key = (in_obj.color, "sig", in_obj.sig)
                        elif match_type == "d4_invariant":
                            rule_key = (in_obj.color, "sig_d4", in_obj.sig_d4)
                        else:
                            # IoUベース: 色のみでルールを定義（最も汎用的）
                            rule_key = (in_obj.color, None, None)

                        # 矛盾するルールの検出
                        if rule_key in rules and rules[rule_key] != out_obj.color:
                            ok = False; break
                        rules[rule_key] = out_obj.color

                if not ok: break
            except Exception:
                # 合成中のエラーは失敗とみなす
                ok = False; break

        if not ok or not rules:
            return [], 0.0

        # DSLプログラムの生成
        prog = []
        for (t_color, sig_type, sig_val), n_color in rules.items():
            step = {"op": "recolor", "target_color": t_color, "new_color": n_color}
            if sig_type == "sig":
                step["match_sig"] = sig_val
            elif sig_type == "sig_d4":
                step["match_sig_d4"] = sig_val
            prog.append(step)

        # 信頼度の計算
        if final_match_type == "exact":
            confidence = 0.98
        elif final_match_type == "d4_invariant":
            confidence = 0.92
        elif final_match_type == "iou_based":
            confidence = 0.85 # IoU閾値に依存
        else:
            confidence = 0.0

        return prog, confidence


    def solve(self, task: ARCTask, budget: float, safe_mode: bool) -> List[CandidateSolution]:
        # safe_modeが有効な場合はスキップ
        if safe_mode:
             return []

        # ENABLE_INDUCTIONフラグは_match_objects内でD4試行の可否判断に使用される

        try:
            prog, score = self._synthesize_program(task)
        except Exception:
            return []

        if not prog or score == 0.0: return []

        # ソルバー関数の生成
        def solver(g: Grid, p=prog, a=self.abstraction, d=self.dsl) -> Grid:
            try:
                scene = a.abstract(g)
                new_scene = d.execute(p, scene)
                return new_scene.to_grid()
            except Exception:
                # 実行時エラー発生時は入力をそのまま返す（安全性重視）
                return grid_to_numpy(g)

        source_name = f"Induction-Program(Score:{score:.2f})"
        return [CandidateSolution(source_name, score, solver)]

# --- Engine 2: Transduction (Heuristics) --------------------------------------
class KUTTransductionEngine:
    # タスクタイプを分析し、特化型のヒューリスティクスを適用
    def __init__(self, am: AbstractionModule):
        self.abstraction = am

    def analyze_task_type(self, task: ARCTask) -> Dict[str, float]:
        # タスクタイプの分析（色変更、穴埋め、移動など）
        # [最適化] 移動タスク判定の精度向上 (Verified Translation)
        scores = {'color': 0.1, 'fill': 0.1, 'move': 0.1}
        if not task.train_pairs or isinstance(np, DummyNp): return scores

        move_analysis = [] # (dr, dc, consistency_score)

        for p in task.train_pairs:
            i, o = p['input'], p['output']
            shape_i, shape_o = getattr(i, 'shape', None), getattr(o, 'shape', None)

            try:
                sum_i_pos = np.count_nonzero(i)
                sum_o_pos = np.count_nonzero(o)
            except: sum_i_pos, sum_o_pos = 0, 0

            # 穴埋めタスクの判定
            if sum_o_pos > sum_i_pos:
                scores['fill'] += 0.3 # 0.2 -> 0.3

            # 色変更タスクの判定
            if shape_i != shape_o or np.any(i != o):
                scores['color'] += 0.1

            # --- 移動タスク判定ロジック強化 (Verified Translation) ---
            # 条件: 形状一致、非ゼロピクセル数一致
            if shape_i == shape_o and sum_i_pos == sum_o_pos and sum_i_pos > 0:
                try:
                    # 背景色(0)以外の座標を取得 (背景色動的検出はsolveで行うため、ここでは簡易的に0以外とする)
                    i_c, o_c = np.argwhere(i > 0), np.argwhere(o > 0)

                    if len(i_c) == len(o_c) and len(i_c) > 0:
                        # 重心の移動ベクトル計算
                        mean_i = np.mean(i_c, axis=0)
                        mean_o = np.mean(o_c, axis=0)
                        dr, dc = mean_o[0] - mean_i[0], mean_o[1] - mean_i[1]

                        # 整数移動量に丸める
                        dr_int, dc_int = int(round(dr)), int(round(dc))

                        # 移動量を適用して一致度を検証
                        H, W = shape_i
                        moved_i = np.zeros_like(i)
                        # 安全なシフト範囲計算
                        r_start_i, r_end_i = max(0, -dr_int), min(H, H - dr_int)
                        c_start_i, c_end_i = max(0, -dc_int), min(W, W - dc_int)
                        r_start_m, r_end_m = max(0, dr_int), min(H, H + dr_int)
                        c_start_m, c_end_m = max(0, dc_int), min(W, W + dc_int)

                        # シフト実行
                        if r_start_i < r_end_i and c_start_i < c_end_i:
                             moved_i[r_start_m:r_end_m, c_start_m:c_end_m] = i[r_start_i:r_end_i, c_start_i:c_end_i]

                        # 出力との一致度（精度）
                        consistency = np.sum(moved_i == o) / i.size
                        move_analysis.append((dr_int, dc_int, consistency))
                    else:
                        move_analysis.append((0, 0, 0.0))
                except Exception:
                     move_analysis.append((0, 0, 0.0))
            else:
                move_analysis.append((0, 0, 0.0))

        # 移動分析結果の集約
        if move_analysis:
            # 全ペアで移動ベクトルが一致し、かつ一貫性スコアが高いか
            vectors = set((m[0], m[1]) for m in move_analysis)
            avg_consistency = sum(m[2] for m in move_analysis) / len(move_analysis)

            # 非常に一貫性が高い場合(98%以上)、移動タスクと断定
            if len(vectors) == 1 and avg_consistency >= 0.98:
                scores['move'] = 0.95
            elif avg_consistency > 0.8:
                scores['move'] = 0.6 # 中程度の信頼度

        return scores

    def solve(self, task: ARCTask, budget: float, heuristic_scores: Dict[str, float], safe_mode: bool) -> List[CandidateSolution]:
        # 依存関係チェック
        if scipy_label is None or scipy_binary_fill_holes is None or isinstance(np, DummyNp): return []

        cands = []
        MAX_CAND = CONFIG.AUTOTUNE_CFG.get("MAX_CAND_PER_OP", 3)

        # H1: 最大オブジェクトの色変更 (T-Color)
        try:
            # [最適化] 抽象化を利用し、背景色を考慮してターゲットカラーを特定
            c_c = Counter()
            for p in task.train_pairs:
                if p['output'].size > 0:
                    out_s = self.abstraction.abstract(p['output'])
                    bg_color = out_s.background_color

                    # 背景色以外のピクセルを集計
                    out_fg = p['output'][p['output'] != bg_color]
                    if out_fg.size > 0:
                        c_c.update(out_fg.flatten())

            t_c = -1
            if c_c:
                t_c = c_c.most_common(1)[0][0]

            def s_c(g:Grid, c=t_c)->Grid:
                if c == -1 or g.size == 0: return grid_to_numpy(g)
                try:
                    # 背景色を考慮してフォアグラウンドを抽出
                    s = self.abstraction.abstract(g)
                    bg = s.background_color
                    fg_mask = (g != bg)

                    # 最大の連結成分を特定
                    structure = np.ones((3, 3), dtype=int)
                    l, n = scipy_label(fg_mask, structure=structure)
                    if n == 0: return grid_to_numpy(g)

                    # [最適化] 堅牢性強化: bincountとサイズの安全チェック
                    sizes = np.bincount(l.ravel())
                    # 背景(ラベル0)を除いたサイズで最大のものを探す
                    if len(sizes) > 1 and sizes[1:].size > 0:
                        # argmaxはインデックス0から始まるため+1する
                        b_l = sizes[1:].argmax() + 1
                        # 色を変更
                        ng = g.copy()
                        ng[l == b_l] = c
                        return ng
                except Exception:
                     pass
                return grid_to_numpy(g)

            if len(cands) < MAX_CAND and t_c != -1:
                conf = max(0.55, min(0.85, 0.55 + heuristic_scores['color'] * 0.3))
                cands.append(CandidateSolution("T-Color", conf, s_c))
        except Exception: pass

        # H2: 穴埋め (T-Fill)
        try:
            is_f = heuristic_scores['fill'] > 0.3 # 閾値調整
            f_c = -1

            # 塗りつぶし色の特定（出力が単色のフォアグラウンドの場合に限定）
            if is_f and task.train_pairs:
                 p = task.train_pairs[0]
                 if p['output'].size > 0:
                    out_s = self.abstraction.abstract(p['output'])
                    bg = out_s.background_color
                    out_fg = p['output'][p['output'] != bg]

                    if out_fg.size > 0:
                        co = np.unique(out_fg)
                        if len(co) == 1:
                            f_c = co[0]

            def s_f(g:Grid, a=is_f, c=f_c)->Grid:
                if not a or c == -1 or g.size == 0: return grid_to_numpy(g)
                try:
                    s = self.abstraction.abstract(g)
                    bg = s.background_color
                    fg_mask = (g != bg)

                    structure = np.ones((3, 3), dtype=int)
                    l, n = scipy_label(fg_mask, structure=structure)
                    if n == 0: return grid_to_numpy(g)

                    ng = g.copy()
                    # 各オブジェクトについて穴埋めを実行
                    for i in range(1, n + 1):
                        m = (l == i)
                        f = scipy_binary_fill_holes(m)
                        # 埋められた部分(f & ~m)に色を塗る
                        ng[f & ~m] = c
                    return ng
                except Exception:
                    pass
                return grid_to_numpy(g)

            if is_f and f_c != -1 and len(cands) < MAX_CAND:
                conf = max(0.60, min(0.90, 0.60 + heuristic_scores['fill'] * 0.3))
                cands.append(CandidateSolution("T-Fill", conf, s_f))
        except Exception: pass

        # H3: オブジェクト移動 (T-Move)
        try:
            # [最適化] analyze_task_typeで高精度判定された場合(0.95)のみを採用
            dr, dc, is_m = 0, 0, heuristic_scores['move'] >= 0.95

            if is_m and task.train_pairs:
                # 最初のペアから移動量を確定させる（analyzeの結果を信頼し、移動量を再現計算）
                inp, outp = task.train_pairs[0]['input'], task.train_pairs[0]['output']

                # analyzeでの条件（Verified Translationの前提条件）を再確認
                # ここでは簡易的に非ゼロ要素で比較（背景色に関わらず移動を検出するため）
                if inp.size > 0 and inp.shape == outp.shape and np.count_nonzero(inp) == np.count_nonzero(outp):
                    try:
                        i_c, o_c = np.argwhere(inp>0), np.argwhere(outp>0)
                        if len(i_c) > 0 and len(i_c) == len(o_c):
                             # 重心移動量から整数移動量を推定
                            dr = int(round(np.mean(o_c[:,0]) - np.mean(i_c[:,0])))
                            dc = int(round(np.mean(o_c[:,1]) - np.mean(i_c[:,1])))
                        else: is_m = False
                    except: is_m = False
                else: is_m = False

            def s_m(g:Grid, a=is_m, d_r=dr, d_c=dc) -> Grid:
                if not a or g.size == 0 or (d_r == 0 and d_c == 0): return grid_to_numpy(g)

                try:
                    # [最適化] 安全で効率的な非循環シフト（平行移動）
                    # 背景色を考慮して初期化
                    s = self.abstraction.abstract(g)
                    bg = s.background_color
                    H, W = g.shape
                    ng = np.full_like(g, bg)

                    # 安全なコピー範囲を計算
                    r_src_start, r_src_end = max(0, -d_r), min(H, H - d_r)
                    c_src_start, c_src_end = max(0, -d_c), min(W, W - d_c)
                    r_dst_start, r_dst_end = max(0, d_r), min(H, H + d_r)
                    c_dst_start, c_dst_end = max(0, d_c), min(W, W + d_c)

                    # 有効な範囲がある場合のみコピー
                    if r_src_start < r_src_end and c_src_start < c_src_end:
                        ng[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = g[r_src_start:r_src_end, c_src_start:c_src_end]
                    return ng
                except Exception:
                     pass
                return grid_to_numpy(g)

            if is_m and (dr != 0 or dc != 0) and len(cands) < MAX_CAND:
                # 移動タスクは信頼度が高い
                conf = max(0.90, heuristic_scores['move'])
                cands.append(CandidateSolution("T-Move", conf, s_m))
        except Exception: pass

        return cands

# ==============================================================================
# 6. kutos/engine_kut30.py (KUT30 Collatz Beam - Governed & Optimized)
# ==============================================================================
# (OMUX004o: Ops効率化(Recolorベクトル化, Move改善)、探索空間拡大(Fill追加)、Learner強化(Verified Translation, ベクトル化学習)、ヘルパー最適化)

# --- Engine 3: KUT30 (Semantic Search) ----------------------------------------
import time, gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Callable, Optional
from collections import Counter
import heapq
import statistics
import math # mathは後続のビームサーチで利用されるため維持
import sys # sysは後続のビームサーチで利用されるため維持

# ------------------------------------------------------------------------------
# Optional deps check
# ------------------------------------------------------------------------------
try:
    # [最適化] NumPyの基本機能と効率化に必要な機能(uint8, bincount)を確認
    _NP_OK = not isinstance(np, DummyNp) and hasattr(np, "ndarray") and hasattr(np, "unique") and hasattr(np, "uint8") and hasattr(np, "bincount")
except Exception:
    _NP_OK = False

# ------------------------------------------------------------------------------
# Types
# ------------------------------------------------------------------------------
K30_Pair = Tuple[Grid, Grid]

# ------------------------------------------------------------------------------
# Helpers (K30専用ユーティリティ)
# ------------------------------------------------------------------------------
# [最適化] 効率化と堅牢性の向上

def _np_unavailable() -> bool:
    return not _NP_OK

def k30_hamming(a: Grid, b: Grid) -> int:
    # ハミング距離（異なるピクセル数）
    if _np_unavailable() or not hasattr(a, "shape") or not hasattr(b, "shape"): return 0

    if a.shape == b.shape:
        return int(np.count_nonzero(a != b))

    # [最適化] 形状不一致時の計算を安定化: 共通領域の差分 + サイズ差ペナルティ
    try:
        H_a, W_a = a.shape; H_b, W_b = b.shape
        H_common, W_common = min(H_a, H_b), min(W_a, W_b)

        # 共通部分の差分
        common_diff = np.count_nonzero(a[:H_common, :W_common] != b[:H_common, :W_common])

        # サイズ差ペナルティ（はみ出た部分の面積）
        size_a, size_b = H_a * W_a, H_b * W_b
        penalty = max(size_a, size_b) - (H_common * W_common)

        return int(common_diff + penalty)
    except Exception:
        # 計算失敗時のフォールバック
        sa = getattr(a, "size", 0); sb = getattr(b, "size", 0)
        return max(sa, sb)

def k30_accuracy(a: Grid, b: Grid) -> float:
    # 精度（一致率）
    if _np_unavailable(): return 0.0
    size_a = getattr(a, 'size', 0); size_b = getattr(b, 'size', 0)

    if size_a == 0: return 1.0 if size_b == 0 else 0.0

    # 形状不一致時もk30_hammingを利用（分母は最大サイズ）
    if getattr(a, 'shape', None) != getattr(b, 'shape', None):
        try:
            # max(1, ...) でゼロ除算防止
            return 1.0 - k30_hamming(a, b) / max(1, max(size_a, size_b))
        except Exception: return 0.0

    return 1.0 - k30_hamming(a, b) / max(1, size_a)

def k30_mode_color(g: Grid) -> int:
    # グリッドの最頻色
    # [最適化] np.bincountによる高速化 (uint8グリッド最適化)
    if _np_unavailable() or getattr(g, 'size', 0) == 0: return 0

    try:
        # 高速パス (common.pyでuint8化されている前提)
        if g.dtype == np.uint8:
             counts = np.bincount(g.ravel())
             if counts.size > 0:
                 return int(np.argmax(counts))
    except Exception:
        pass # フォールバックへ

    # フォールバックパス (堅牢性重視)
    try:
        colors, counts = np.unique(g, return_counts=True)
        if colors.size > 0:
            return int(colors[np.argmax(counts)])
    except Exception:
        try:
            # 最終手段（低速だが確実）
            return int(statistics.mode(g.flatten()))
        except Exception:
            pass
    return 0

def k30_connected_components(mask: Grid) -> List["np.ndarray"]:
    # 連結成分の抽出
    if _np_unavailable() or scipy_label is None: return []
    try:
        # 8連結性を明示
        structure = np.ones((3, 3), dtype=int)
        labeled, n = scipy_label(mask, structure=structure)
        return [(labeled == i) for i in range(1, n + 1)]
    except Exception:
        return []

def safe_masks(g: Grid, masks: List["np.ndarray"]) -> List["np.ndarray"]:
    # 空のマスクリストを防ぐ（最低でも全体マスクを返す）
    if _np_unavailable(): return []

    # [最適化] マスクの有効性検証（形状と内容）
    valid_masks = []
    g_shape = getattr(g, 'shape', None)
    if g_shape:
        for m in masks:
            # 形状が一致し、空でないマスクのみを保持
            if hasattr(m, 'shape') and m.shape == g_shape and m.size > 0:
                try:
                    # ブール型保証
                    if m.dtype != bool: m = m.astype(bool)
                    if m.any():
                        valid_masks.append(m)
                except: continue

    if valid_masks:
        return valid_masks

    # 有効なマスクがない場合、全体マスクを生成
    if getattr(g, 'size', 0) > 0:
        try:
            return [np.ones_like(g, dtype=bool)]
        except Exception:
            try:
                if g_shape: return [np.ones(g_shape, dtype=bool)]
            except Exception:
                pass
    return []

def overlay(base: Grid, masks: List["np.ndarray"], preds: List[Grid]) -> Grid:
    # マスクに基づいて予測結果をベースグリッドに重ね合わせる
    if _np_unavailable(): return base
    out = base.copy()
    base_shape = getattr(base, 'shape', None)

    # [最適化] 安全性強化: 形状チェックと型保証
    for m, p in zip(masks, preds):
        # 全てが同じ形状を持つことを確認
        if base_shape and hasattr(m, "shape") and hasattr(p, "shape"):
            if m.shape == base_shape and p.shape == base_shape:
                try:
                    # マスクがブール型であることを保証
                    if m.dtype != bool: m = m.astype(bool)
                    out[m] = p[m]
                except Exception:
                    continue # エラーが発生したマスクはスキップ
    return out

# ------------------------------------------------------------------------------
# Ops (基本操作定義)
# ------------------------------------------------------------------------------
# [最適化] 効率化と探索空間の改善

@dataclass
class K30_Op:
    name: str
    # [最適化] 複雑度(complexity)をfloatで定義し、MDL計算を精緻化
    complexity: float
    apply: Callable[[Grid, Optional["np.ndarray"]], Grid]  # mask-aware

def _ensure_mask(g: Grid, m: Optional["np.ndarray"]) -> "np.ndarray":
    # マスクが存在しない場合は全体マスクを生成し、常にブール型を保証
    if _np_unavailable(): return m
    if m is None:
        try: return np.ones_like(g, dtype=bool)
        except: return np.ones(g.shape, dtype=bool) if hasattr(g, 'shape') else m
    # copy=Falseで効率的にキャスト
    return m.astype(bool, copy=False)

def k30_op_recolor(mapping: Dict[int, int]) -> K30_Op:
    # 色の置換操作
    # [最適化] 高速化: ループベースからベクトル化されたLUT方式へ変更
    mp = {int(k): int(v) for k, v in mapping.items() if int(k) != int(v)}

    def _ap_vectorized(g: Grid, m: Optional["np.ndarray"] = None) -> Grid:
        if _np_unavailable() or not mp or getattr(g, "size", 0) == 0: return g

        try:
            # 1. LUT（ルックアップテーブル）の作成
            max_val = int(g.max())
            # 必要なサイズのLUTを確保
            lut_size = max(max_val + 1, max(mp.keys()) + 1 if mp else 0)
            # LUT初期化（デフォルトは入力値＝出力値）。dtypeはgに合わせる(uint8推奨)。
            lut = np.arange(lut_size, dtype=g.dtype)

            # マッピングをLUTに適用
            keys = np.array(list(mp.keys()), dtype=int)
            values = np.array(list(mp.values()), dtype=g.dtype)

            # LUT範囲外のキーを安全に除外
            valid_keys_mask = (keys < lut_size)
            keys = keys[valid_keys_mask]
            values = values[valid_keys_mask]

            if keys.size == 0: return g

            lut[keys] = values

            # 2. LUTの適用（NumPyの高度なインデックス参照で一括変換）
            transformed = lut[g]

            # 3. マスク処理
            m_local = _ensure_mask(g, m)

            # マスクの形状チェック（_ensure_maskのフォールバック対策）
            if m_local is None or m_local.shape != g.shape:
                 return transformed # マスクが無効なら全体適用として扱う

            if np.all(m_local):
                # 全体マスクならそのまま返す
                return transformed
            else:
                # マスク領域のみ変更を適用
                out = g.copy()
                out[m_local] = transformed[m_local]
                return out
        except Exception:
             # ベクトル化失敗時（例：IndexError, メモリ不足）は安全にフォールバック
             return g

    # 複雑度はマッピング数に応じて微増
    complexity = 1.0 + len(mp) * 0.05
    return K30_Op(f"Recolor({len(mp)})", complexity, _ap_vectorized)

def k30_op_move(dx: int, dy: int, bg: Optional[int] = None) -> K30_Op:
    # マスク領域の移動操作。
    # [最適化] ロジック改善: プルシフトから「True Masked Move」へ変更
    # マスクされた領域が移動し、元の場所は背景色でクリアされる。
    def _ap(g: Grid, m: Optional["np.ndarray"] = None) -> Grid:
        if _np_unavailable() or getattr(g, "size", 0) == 0 or (dx == 0 and dy == 0): return g

        H, W = g.shape
        out = g.copy()
        m_local = _ensure_mask(g, m)

        # 背景色の決定
        b = int(k30_mode_color(g)) if bg is None else int(bg)

        try:
            # マスクされているピクセルの座標を取得
            ys, xs = np.where(m_local)
            if ys.size == 0: return out

            # 移動先の座標を計算
            ys_dst, xs_dst = ys + dy, xs + dx

            # 境界チェック
            valid = (ys_dst >= 0) & (ys_dst < H) & (xs_dst >= 0) & (xs_dst < W)

            # 1. 移動元を背景色でクリア
            out[ys, xs] = b

            # 2. 有効な移動先に値をコピー (元のグリッドgからコピー)
            if np.any(valid):
                # 注意: 移動先が重複する場合、コピー順序によって結果が変わる可能性があるが、
                # ここではNumPyのデフォルト動作に任せる（通常は決定論的）。
                out[ys_dst[valid], xs_dst[valid]] = g[ys[valid], xs[valid]]
        except Exception: return g
        return out
    # 移動操作の複雑度を設定
    return K30_Op(f"Move({dx},{dy})", 1.5, _ap)

def k30_op_mirror_h() -> K30_Op:
    # 水平反転 (グローバル反転してからマスク適用)
    def _ap(g: Grid, m: Optional["np.ndarray"] = None) -> Grid:
        if _np_unavailable() or getattr(g, "size", 0) == 0: return g
        try:
            # 全体を反転
            flipped = g[:, ::-1]
            m_local = _ensure_mask(g, m)

            out = g.copy()
            # マスク内のみ結果を適用
            out[m_local] = flipped[m_local]
        except Exception: return g
        return out
    return K30_Op("MirrorH", 1.2, _ap)

def k30_op_rotate90() -> K30_Op:
    # 90度回転（形状が変わる場合はクリップ/パディング）
    def _ap(g: Grid, m: Optional["np.ndarray"] = None) -> Grid:
        if _np_unavailable() or getattr(g, "size", 0) == 0: return g
        H, W = g.shape
        try:
            rotated = np.rot90(g, k=1)
            rh, rw = rotated.shape

            # 元の形状に収まるようにクリップし、残りを背景色で埋める
            b = k30_mode_color(g)
            tmp = np.full_like(g, b)
            h_min, w_min = min(H, rh), min(W, rw)
            tmp[:h_min, :w_min] = rotated[:h_min, :w_min]

            out = g.copy()
            m_local = _ensure_mask(g, m)
            out[m_local] = tmp[m_local]
        except Exception: return g
        return out
    return K30_Op("Rotate90", 1.5, _ap)

# [最適化] 新規追加操作: 穴埋め
def k30_op_fill(color: int) -> K30_Op:
    # マスク領域内の穴を特定の色で埋める
    def _ap(g: Grid, m: Optional["np.ndarray"] = None) -> Grid:
        if _np_unavailable() or scipy_binary_fill_holes is None or getattr(g, "size", 0) == 0: return g
        out = g.copy()
        m_local = _ensure_mask(g, m)

        try:
            # マスク領域で穴埋めを実行
            filled = scipy_binary_fill_holes(m_local)
            # 埋められた部分（filled かつ not m_local）に色を塗る
            holes = filled & (~m_local)
            if np.any(holes):
                out[holes] = int(color)
        except Exception: return g
        return out
    return K30_Op(f"Fill({color})", 1.3, _ap)


def k30_op_stub(name: str) -> K30_Op:
    return K30_Op(f"{name}[stub]", 1.0, lambda g, m=None: g)

# ------------------------------------------------------------------------------
# Learners (学習器)
# ------------------------------------------------------------------------------
def k30_learn_global_recolor(train: List[K30_Pair]) -> Dict[int, int]:
    # 学習ペアからグローバルな色のマッピングを学習
    # [最適化] NumPyベクトル化による効率改善
    if _np_unavailable(): return {}
    counts: Dict[int, Counter] = {}

    for ain, aout in train:
        if getattr(ain, 'size', 0) == 0 or getattr(aout, 'size', 0) == 0: continue

        # 形状が異なる場合は共通部分のみ使用
        H = min(ain.shape[0], aout.shape[0]); W = min(ain.shape[1], aout.shape[1])
        ain_c, aout_c = ain[:H, :W], aout[:H, :W]

        try:
            # 高速パス: フラット化してペアを作成し、一括で集計
            # 1. 入出力ペアの作成
            pairs = np.vstack((ain_c.ravel(), aout_c.ravel())).T
            # 2. 変化がないペアを除外（効率化）
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]

            if pairs.size == 0: continue

            # 3. ユニークなペアとその出現回数をカウント
            unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)

            # 4. 結果をcounts辞書に集約
            for (src, dst), count in zip(unique_pairs, pair_counts):
                counts.setdefault(int(src), Counter()).update({int(dst): int(count)})

        except Exception:
            # フォールバックパス（低速だが安全）
            for v in np.unique(ain_c):
                iv = int(v)
                mask = (ain_c == iv)
                tv, tc = np.unique(aout_c[mask], return_counts=True)
                counts.setdefault(iv, Counter()).update({int(tvv): int(tcc) for tvv, tcc in zip(tv, tc)})

    mapping: Dict[int, int] = {}
    for v, ctr in counts.items():
        # 最も頻度の高い対応色を採用
        if ctr:
            target_color = int(ctr.most_common(1)[0][0])
            # 変化がない場合はマッピングに追加しない（高速パスでは不要だがフォールバックで必要）
            if v != target_color:
                mapping[v] = target_color
    return mapping

def k30_learn_move(train: List[K30_Pair]) -> Tuple[int, int]:
    # 学習ペアからグローバルな移動量を学習
    # [最適化] Verified Translationロジック導入 (精度向上)
    if _np_unavailable(): return (0, 0)

    move_analysis = [] # (dr, dc, consistency_score)

    for ain, aout in train:
        if getattr(ain, 'size', 0) == 0 or getattr(aout, 'size', 0) == 0: continue

        # 前提条件: 形状一致、非ゼロピクセル数一致（簡易判定）
        if ain.shape != aout.shape or np.count_nonzero(ain) != np.count_nonzero(aout) or np.count_nonzero(ain) == 0:
             move_analysis.append((0, 0, 0.0)); continue

        try:
            # 背景色(0)以外の座標を取得（簡易的に0以外とする。背景色が0でない場合も機能するが精度は低下する可能性あり）
            i_c, o_c = np.argwhere(ain > 0), np.argwhere(aout > 0)

            # 1. 重心の移動ベクトルから移動量を推定
            mean_i, mean_o = np.mean(i_c, axis=0), np.mean(o_c, axis=0)
            dr_int = int(round(mean_o[0] - mean_i[0]))
            dc_int = int(round(mean_o[1] - mean_i[1]))

            # 2. 推定した移動量を適用して一致度を検証 (Verified Translation)
            H, W = ain.shape
            moved_i = np.zeros_like(ain)
            # 安全なシフト範囲計算（ソースとデスティネーション）
            r_si, r_ei = max(0, -dr_int), min(H, H - dr_int) # Source Input
            c_si, c_ei = max(0, -dc_int), min(W, W - dc_int)
            r_sm, r_em = max(0, dr_int), min(H, H + dr_int)  # Source Moved (Destination)
            c_sm, c_em = max(0, dc_int), min(W, W + dc_int)

            # シフト実行
            if r_si < r_ei and c_si < c_ei:
                 moved_i[r_sm:r_em, c_sm:c_em] = ain[r_si:r_ei, c_si:c_ei]

            # 出力との一致度（精度）を計算
            consistency = np.sum(moved_i == aout) / ain.size
            # (dx, dy)形式で格納
            move_analysis.append((dc_int, dr_int, consistency))

        except Exception:
            move_analysis.append((0, 0, 0.0))

    if not move_analysis: return (0, 0)

    # 95%以上の一貫性を持つ移動ベクトルのみを候補とする
    consistent_moves = Counter()
    for dx, dy, consistency in move_analysis:
        # 移動量が0でなく、かつ高精度なもの
        if (dx != 0 or dy != 0) and consistency >= 0.95:
            consistent_moves[(dx, dy)] += 1

    if not consistent_moves: return (0, 0)

    # 最も頻度の高い一貫した移動ベクトルを採用
    best_move, count = consistent_moves.most_common(1)[0]

    # 全体の半分以上のペアで確認できた場合のみ採用（信頼性確保）
    if count >= len(train) / 2:
        return best_move
    return (0, 0)

# ------------------------------------------------------------------------------
# Consistency & Collatz (Patched: MDL weight is tunable)
# ------------------------------------------------------------------------------
class K30_ConsistencyChecker:
    def __init__(self, train: List[K30_Pair], mdl_weight: float = 0.02):
        self.train = train
        self.mdl_weight = float(mdl_weight)  # ← 動的に変える

    def set_mdl_weight(self, w: float):
        try:
            w = float(w)
            self.mdl_weight = max(0.0, w)
        except Exception:
            return

    def score(self, program: Callable[[Grid], Grid], complexity: float) -> Dict[str, float]:
        # プログラムの精度と複雑さに基づくスコア
        # [最適化] complexityをfloatで受け取る
        accs: List[float] = []
        for ain, aout in self.train:
            try:
                accs.append(k30_accuracy(program(ain), aout))
            except Exception:
                # 実行時エラーは精度0
                accs.append(0.0)

        train_acc = float(statistics.mean(accs)) if accs else 0.0

        # ペナルティ計算 (複雑度が1.0を超える部分に重みを掛ける)
        penalty = self.mdl_weight * max(0.0, complexity - 1.0)
        final = train_acc - penalty
        return {"train_acc": train_acc, "penalty": penalty, "final": final}

def k30_collatz_sequence(n: int, limit: int = 64) -> List[int]:
    # （既存のまま）コラッツ列
    s = [n]
    while n > 1 and len(s) < limit:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s.append(n)
    return s

# ------------------------------------------------------------------------------
# Clustering (Semantic Segmentation)
# ------------------------------------------------------------------------------
class K30_SemanticCluster:
    # セマンティッククラスタリング（色または連結性に基づく分割）
    def __init__(self, criterion: str = 'color'):
        self.criterion = criterion

    def split(self, g: Grid) -> List["np.ndarray"]:
        if _np_unavailable() or getattr(g, 'size', 0) == 0: return []

        if self.criterion == 'color':
            # 色ごとに分割
            try:
                unique_colors = np.unique(g)
                # [最適化] 色の順序を固定することで決定性を確保
                unique_colors.sort()
                return [(g == c) for c in unique_colors]
            except Exception:
                return []
        else:
            # 連結成分ごとに分割
            # [最適化] 背景色を正しく除外する
            bg = k30_mode_color(g)
            foreground_mask = (g != bg)
            components = k30_connected_components(foreground_mask)
            # SciPyが失敗した場合や前景がない場合のフォールバック
            if not components and np.any(foreground_mask):
                 return [foreground_mask]
            return components
# ------------------------------------------------------------------------------
# Collatz Beam (KUT30 Governed Beam Search, Alpha–Collatz fused) [KUT30 Update]
# ------------------------------------------------------------------------------
# [最適化] K30にAlpha-Collatz Fusionを統合し、探索効率を最大化
class K30_CollatzBeam:
    def __init__(self, checker: K30_ConsistencyChecker, K: int,
                 monitor: ResourceMonitor = None, governor: BeamGovernor = None,
                 time_budget: float = float('inf')):
        self.checker = checker
        # Governorの最小ビーム幅設定と整合性を取る
        self.K_default = max(getattr(governor, 'min_beam', 4), int(K))
        self.monitor = monitor
        self.governor = governor
        self.time_budget = time_budget
        self.start_time = time.time()

    # --- fingerprint → (α, s0) [Alpha-Collatz Fusion Logic] ------------------
    @staticmethod
    def _feat_from_train(train):
        # タスクの特徴量（色数、連結成分数、サイズ、対称性）を抽出
        if _np_unavailable() or not train: return (6, 8, 20, 20, 0.3)

        def _components(g):
            # 背景色以外の連結成分数を計算
            if scipy_label is None: return 8
            try:
                bg = k30_mode_color(g)
                mask = (g != bg)
                # k30_connected_componentsは内部でscipy_labelを呼ぶ
                return len(k30_connected_components(mask))
            except Exception: return 8

        try:
            g = train[0][0]
            H, W = int(getattr(g, "shape", (0,0))[0]), int(getattr(g, "shape", (0,0))[1])
        except Exception:
            return (6, 8, 20, 20, 0.3)

        try: colors = int(np.unique(g).size)
        except Exception: colors = 6
        components = int(_components(g))

        try: symmetry = float((g == g[:, ::-1]).mean())
        except Exception: symmetry = 0.3
        return (max(1,colors), max(1,components), max(1,H), max(1,W), symmetry)

    @staticmethod
    def _alpha_from_feat(colors, components, H, W, symmetry, base=1.0):
        # 特徴量からタスクの複雑度αを推定 (0.7～1.5)
        # KUT30向けパラメータ調整: componentsの寄与をやや高めに設定
        chi = 0.35*math.log1p(colors) + 0.45*math.log1p(components) + 0.20*math.log1p(H*W) - 0.2*symmetry
        a = 0.7 + 0.5*chi
        # 平均的な複雑度(chi≈1.0)でα≈1.2となるように調整
        return max(0.7, min(1.5, base * a / 1.0))

    @staticmethod
    def _s0_from_feat(colors, components, H, W, symmetry, scale=1.0):
        # 特徴量からコラッツ数列の初期値s0を決定 (7～63の奇数)
        chi = 0.35*math.log1p(colors) + 0.45*math.log1p(components) + 0.20*math.log1p(H*W) - 0.2*symmetry
        s0 = int(round(40*chi) + 5)
        # KUT30は探索深度が必要なため、scale=1.0をデフォルトとする
        s0 = max(7, min(63, int(round(s0 * scale))))
        if s0 % 2 == 0: s0 += 1
        return s0

    @staticmethod
    def _collatz_seq(s0:int, max_steps:int=8):
        # コラッツ数列と位相（奇数/偶数）を生成
        s, seq = s0, []
        for _ in range(max_steps):
            phase = 'odd' if (s % 2 == 1) else 'even'
            seq.append((phase, s))
            if s == 1: break
            s = 3*s + 1 if phase == 'odd' else s // 2
        return seq

    @staticmethod
    def _fused_mult(alpha_eff: float, phase: str):
        # αと位相に基づいてKとMDLの倍率を決定
        # KUT30: Kはα^1.25に比例。奇数で拡張(1.20)、偶数で圧縮(0.70)。
        K_mult = (alpha_eff ** 1.25) * (1.20 if phase == 'odd' else 0.70)
        # MDLは1/αに比例。奇数で緩和(0.85)、偶数で強化(1.25)。
        MDL_mult = (1.0 / alpha_eff) * (0.85 if phase == 'odd' else 1.25)
        return K_mult, MDL_mult
    # --------------------------------------------------------------------------

    def derive_budget(self) -> int:
        # タスクの複雑さ（初期エラー量）から探索予算（コラッツN0）を決定
        errs = [k30_hamming(ain, aout) for ain, aout in self.checker.train]
        if not errs: return 7
        E0 = int(statistics.mean(errs))
        # [最適化] 12Hルール対応: 探索予算範囲を拡大 (7-27 -> 7-35)
        return max(7, min(35, E0 // 8 + 7))

    def _program_from_sequences(self,
                                masks_fn: Callable[[Grid], List["np.ndarray"]],
                                seqs: List[List[K30_Op]]) -> Callable[[Grid], Grid]:
        # 操作シーケンスのリストから実行可能なプログラムを生成
        def _prog(g: Grid) -> Grid:
            if _np_unavailable(): return g
            # 実行時にクラスタリング（マスク生成）
            masks = safe_masks(g, masks_fn(g))
            # grid_to_numpyで安全なコピーを作成
            if not masks: return grid_to_numpy(g)

            C = min(len(masks), len(seqs))
            if C == 0: return grid_to_numpy(g)

            masks_use = masks[:C]
            seqs_use = seqs[:C]
            preds: List[Grid] = []

            # 各クラスタ（マスク）に対して対応する操作シーケンスを適用
            for m, seq in zip(masks_use, seqs_use):
                sub = g.copy()
                for op in seq:
                    try:
                        sub = op.apply(sub, m)
                    except Exception:
                        # 操作失敗時はそのステップをスキップ（堅牢性重視）
                        pass
                preds.append(sub)
            # 結果を合成
            return overlay(g, masks_use, preds)
        return _prog

    def search(self, base_ops: List[K30_Op],
               masks_fn: Callable[[Grid], List["np.ndarray"]],
               train: List[K30_Pair]) -> Tuple[Callable[[Grid], Grid], Dict[str, float]]:

        if _np_unavailable() or not train:
            return (lambda g: g), {"final": 0.0}

        # --- 初期化フェーズ ---

        # 1. Alpha-Collatz Initialization
        colors, components, H, W, sym = self._feat_from_train(train)
        alpha0 = self._alpha_from_feat(colors, components, H, W, sym, base=1.0)
        s0     = self._s0_from_feat(colors, components, H, W, sym, scale=1.0)
        seq    = self._collatz_seq(s0, max_steps=8)

        # 2. Beam Initialization
        n_clusters = len(safe_masks(train[0][0], masks_fn(train[0][0])))
        if n_clusters == 0: n_clusters = 1

        def mk_empty(C: int) -> List[List[K30_Op]]:
            return [[] for _ in range(C)]

        prog0 = self._program_from_sequences(masks_fn, mk_empty(n_clusters))
        # [最適化] Complexityはfloatで管理
        sc0 = self.checker.score(prog0, 0.0)["final"]
        # Beam要素: (シーケンスリスト, スコア, 複雑度(float))
        beam: List[Tuple[List[List[K30_Op]], float, float]] = [(mk_empty(n_clusters), sc0, 0.0)]

        # 3. 探索スケジュール (コラッツN0)
        N0 = self.derive_budget()
        # [最適化] 最大ステップ数を拡大 (24 -> 32)
        lengths = [1] + [max(1, n % 5 + 1) for n in k30_collatz_sequence(N0)][:32]

        current_K = int(self.K_default)
        base_mdl = float(getattr(self.checker, "mdl_weight", 0.02))

        # --- ビームサーチのメインループ ---
        for step_idx, L in enumerate(lengths):

            # --- 動的調整フェーズ (α×Collatz + Governor) ---

            # 1. α×Collatzによる位相調整
            phase, sval = seq[min(step_idx, len(seq)-1)]
            # αの実効値を位相に応じて変動させる (奇数で強調、偶数で緩和)。範囲を微調整(1.8)。
            alpha_eff = max(0.6, min(1.8, alpha0 * (1.20 if phase == 'odd' else 0.85)))

            K_mult, MDL_mult = self._fused_mult(alpha_eff, phase)
            # 位相に応じたKとMDLを計算
            phase_K = max(4, int(round(self.K_default * K_mult)))
            self.checker.set_mdl_weight(base_mdl * MDL_mult)

            # 2. Governorによるリソース調整 (MemGuard統合)
            if self.governor and self.monitor:
                try:
                    snap = self.monitor.snapshot()
                    elapsed = time.time() - self.start_time
                    # Governorのデフォルト値として位相Kを使用
                    default_K = min(phase_K, self.governor.max_beam)

                    k_suggest, _, reason = self.governor.suggest(snap, default_K, default_K, elapsed, self.time_budget)

                    if isinstance(k_suggest, int) and k_suggest > 0:
                         # ビーム幅変更時のログ出力
                        if k_suggest != current_K:
                            ramf_log = float(getattr(snap, "ram_frac")() if hasattr(snap, "ram_frac") else 0.0)
                            vramfs_log = snap.vram_fracs()
                            vramf_log = max(vramfs_log) if vramfs_log else 0.0
                            # [最適化] ログに位相情報を追加
                            print(f"[KUT30-Gov] {time.strftime('%H:%M:%S')} | Phase:{phase}({step_idx+1}/{len(lengths)}) α:{alpha_eff:.2f} | Time: {elapsed:.1f}s/{self.time_budget:.0f}s | Beam: {current_K} -> {k_suggest} (Target:{phase_K}) | Reason: {reason} | RAM: {ramf_log*100:.1f}% VRAM: {vramf_log*100:.1f}%", file=sys.stderr)

                        current_K = max(self.governor.min_beam, min(self.governor.max_beam, k_suggest))
                    else:
                        # Governorが無効な提案をした場合は位相Kに戻す
                        current_K = phase_K

                    # 緊急対応（ハードリミット超過時）
                    ramf_check = float(getattr(snap, "ram_frac")() if hasattr(snap, "ram_frac") else 0.0)
                    # Governorのハードリミット設定（デフォルト0.88）を参照
                    HARD_RAM_LIMIT = getattr(self.governor, "hard_ram", 0.88)
                    if ramf_check > HARD_RAM_LIMIT:
                        current_K = max(self.governor.min_beam, max(4, current_K // 3)) # より積極的に削減
                        gc.collect()
                        print(f"[KUT30-Gov] EMERGENCY: High RAM usage ({ramf_check*100:.1f}% > {HARD_RAM_LIMIT*100:.1f}%). Beam reduced to {current_K}. GC called.", file=sys.stderr)

                    # ビームの刈り込み
                    if len(beam) > current_K:
                        beam = beam[:current_K]

                except Exception as e:
                    print(f"[KUT30-Governor] Governor error: {e}", file=sys.stderr)
                    current_K = phase_K # エラー時は位相Kに戻す
            else:
                # Governor/Monitorが無効な場合は位相Kを使用
                current_K = phase_K

            # --- 探索フェーズ ---

            # ビームの拡張
            newbeam: List[Tuple[List[List[K30_Op]], float, float]] = []
            for seqs, sc, comp in beam:
                # 各クラスタに対して、各基本操作を適用して新しいシーケンスを生成
                for cidx in range(n_clusters):
                    for op in base_ops:
                        nseqs = [s[:] for s in seqs]
                        nseqs[cidx] = nseqs[cidx] + [op]

                        # 新しいプログラムを評価
                        prog = self._program_from_sequences(masks_fn, nseqs)
                        # [最適化] complexity(float)を加算
                        new_comp = comp + op.complexity
                        # 現在のMDL Weightでスコア計算
                        score = self.checker.score(prog, new_comp)
                        newbeam.append((nseqs, score["final"], new_comp))

            if not newbeam:
                break

            # ビームの更新（上位K個を保持）
            if len(newbeam) > current_K:
                # heapqを使用して効率的に上位K個を選択
                beam = heapq.nlargest(current_K, newbeam, key=lambda t: t[1])
            else:
                newbeam.sort(key=lambda t: t[1], reverse=True)
                beam = newbeam

        # --- 結果集約フェーズ ---
        if not beam:
            return (lambda g: grid_to_numpy(g)), {"final": 0.0}

        # 最終評価前にMDL Weightを元に戻す
        self.checker.set_mdl_weight(base_mdl)

        best_seqs, best_score, best_comp = max(beam, key=lambda t: t[1])
        best_prog = self._program_from_sequences(masks_fn, best_seqs)
        # 最終的なメトリクスは基本MDLで再計算
        metrics = self.checker.score(best_prog, best_comp)
        # 探索中に見つかった最高スコアと再計算後のスコアの高い方を採用
        metrics["final"] = float(max(best_score, metrics["final"]))

        return best_prog, metrics

# ------------------------------------------------------------------------------
# Engine Interface
# ------------------------------------------------------------------------------

# [最適化] 穴埋め操作(Fill)の学習ヘルパー
def _k30_learn_fill_color(train: List[K30_Pair]) -> Optional[int]:
    # 学習ペアから一貫した穴埋め色を学習
    if _np_unavailable() or scipy_binary_fill_holes is None: return None
    fill_colors = Counter()
    consistent = True

    for ain, aout in train:
         if ain.shape != aout.shape: continue # 形状変化は対象外

         try:
             # 入力の背景色を推定
             bg = k30_mode_color(ain)

             # 入力の穴領域を特定
             mask_in = (ain != bg)
             filled_in = scipy_binary_fill_holes(mask_in)
             holes_in = filled_in & (~mask_in)

             if not np.any(holes_in): continue

             # 出力でその穴が埋められているか確認 (簡易的に入力の背景色で判断)
             mask_out = (aout != bg)
             filled_out = scipy_binary_fill_holes(mask_out)
             holes_out = filled_out & (~mask_out)

             # 埋められた領域 = (入力の穴) かつ (出力の穴でない)
             filled_area = holes_in & (~holes_out)

             if np.any(filled_area):
                 # 埋められた領域の色を取得
                 colors = np.unique(aout[filled_area])
                 # 単一色で埋められている場合のみ記録
                 if len(colors) == 1:
                     fill_colors[int(colors[0])] += 1
                 else:
                     # 複数色で埋められている場合は一貫性なし
                     consistent = False; break
         except Exception:
             consistent = False; break

    if consistent and fill_colors:
        # 最も頻出する埋め色を採用し、全ペアの半分以上で確認できた場合
        most_common, count = fill_colors.most_common(1)[0]
        if count >= len(train) * 0.5:
            return most_common
    return None


class KUT30SemanticCollatzBeamEngine:
    def __init__(self, criterion: str = 'color', K: int = CONFIG.KUT30_BEAM_K,
                 monitor: ResourceMonitor = None, governor: BeamGovernor = None,
                 time_budget: float = float('inf')):
        self.clusterer = K30_SemanticCluster(criterion)
        self.K = int(K)
        self.monitor = monitor
        self.governor = governor
        self.time_budget = time_budget
        self.best_prog: Optional[Callable[[Grid], Grid]] = None
        self.metrics: Optional[Dict[str, float]] = None
        self.base_ops: List[K30_Op] = []

    def _learn_base_ops(self, train: List[K30_Pair]) -> List[K30_Op]:
        # 学習データから有効な基本操作を学習・選択
        # [最適化] 探索空間の拡大: 穴埋め(Fill)操作の追加
        ops: List[K30_Op] = []

        # 1) リカラー (global mapping)
        mp = k30_learn_global_recolor(train)
        if mp: ops.append(k30_op_recolor(mp))

        # 2) 移動 (Verified Translation)
        dx, dy = k30_learn_move(train)
        if (dx, dy) != (0, 0): ops.append(k30_op_move(dx, dy))

        # 3) 穴埋め (Fill)
        fill_color = _k30_learn_fill_color(train)
        if fill_color is not None:
             # k30_op_fillが定義されていることを確認（先行ステップで定義済み）
             if 'k30_op_fill' in globals():
                 ops.append(globals()['k30_op_fill'](fill_color))

        # 4) 基本的な幾何学変換
        ops.append(k30_op_mirror_h())
        ops.append(k30_op_rotate90())

        # フォールバック（操作が一つもない場合）
        if not ops and not _np_unavailable() and train:
            base = train[0][0]
            bg = k30_mode_color(base)
            try:
                # 背景色へのリカラーマップを作成
                recolor_map = {int(c): int(bg) for c in np.unique(base) if int(c) != bg}
            except Exception:
                recolor_map = {}
            if recolor_map:
                ops.append(k30_op_recolor(recolor_map))
            else:
                ops.append(k30_op_stub("Fallback"))

        # 重複除去
        uniq: Dict[str, K30_Op] = {}
        for o in ops: uniq[o.name] = o
        return list(uniq.values())

    def fit(self, train_grids: List[K30_Pair]):
        if _np_unavailable():
            # 実行環境の安全チェック
            raise RuntimeError("NumPy backend unavailable or incomplete. KUT30 cannot run safely.")

        checker = K30_ConsistencyChecker(train_grids)
        self.base_ops = self._learn_base_ops(train_grids)

        # ビームサーチ実行 (Alpha-Collatz Fused Beam)
        beam = K30_CollatzBeam(checker, K=self.K, monitor=self.monitor,
                               governor=self.governor, time_budget=self.time_budget)
        self.best_prog, self.metrics = beam.search(self.base_ops, self.clusterer.split, train_grids)
        return self

# ------------------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------------------
def make_kut30_candidate_from_arctask(task: ARCTask, criterion: str = 'color',
                                      monitor: ResourceMonitor = None, governor: BeamGovernor = None,
                                      time_budget: float = float('inf')) -> Tuple[str, float, Callable[[Grid], Grid]]:

    # train_pairsはARCTask初期化時にすでにNumPy配列(uint8)に変換済み
    train_pairs: List[K30_Pair] = [(p['input'], p['output']) for p in task.train_pairs]

    # CONFIGからビーム幅Kを取得（AutoTuneで決定された値）
    default_K = getattr(CONFIG, "KUT30_BEAM_K", 18)

    try:
        engine = KUT30SemanticCollatzBeamEngine(
            criterion=criterion, K=default_K,
            monitor=monitor, governor=governor, time_budget=time_budget
        ).fit(train_pairs)
    except Exception as e:
        print(f"[KUT30 Factory] Error during fit: {e}", file=sys.stderr)
        return "KUT30[Error]", 0.0, lambda g: grid_to_numpy(g)


    prog = engine.best_prog
    m = engine.metrics or {"final": 0.0}

    if prog is None or m.get('final', 0.0) <= 0.0:
        # [最適化] 失敗時は安全な恒等変換を返す
        return "KUT30[None]", 0.0, lambda g: grid_to_numpy(g)

    # 信頼度の計算（0.60から0.98の範囲）。最終スコア(MDL適用後)に基づく。
    conf = float(max(0.60, min(0.98, m.get('final', 0.0) * 0.96))) # 係数微調整
    return "KUT30", conf, prog

# ==============================================================================
# 7. kutos/engine_kut32.py (KUT32 Shape & Layout)
#    グリッド全体の形状変化（スケーリング、タイリング）に特化
# ==============================================================================
# (OMUX4o: Alpha-Collatz Fusion Beam 移植済 + 安全性・効率性強化)

# --- Engine 4: KUT32 (Shape Transformation) -----------------------------------

# ---- KUT32 Configuration ----
# メモリ安全性に関する設定
# [最適化] MEM_SAFETYを0.35から0.55へ引き上げ。利用可能メモリの55%まで許容。
MEM_SAFETY = float(os.environ.get("K32_MEM_SAFETY", "0.55"))
# 最大ピクセル数 30M -> 45M (uint8化による余裕を考慮)
HARD_CAP_PX = int(os.environ.get("K32_HARD_CAP_PX", "45000000"))

# ---- KUT32 Helpers ----

# [最適化] _to_ndarrayを削除。以降はcommon.pyのgrid_to_numpyを使用する。

def _mem_ok(shape: Tuple[int, int], dtype = None, safety: float = MEM_SAFETY, hard_cap_px: int = HARD_CAP_PX) -> bool:
    # メモリ使用量が安全範囲内かチェック
    if isinstance(np, DummyNp): return True
    H, W = int(shape[0]), int(shape[1])
    if H <= 0 or W <= 0: return False

    # オーバーフロー防止
    try:
        total_px = H * W
    except OverflowError:
        return False

    if hard_cap_px and total_px > hard_cap_px: return False

    # [最適化] dtypeが指定されていない場合はuint8をデフォルトとする
    if dtype is None:
        dtype = np.uint8 if hasattr(np, 'uint8') else int

    try: itemsize = np.dtype(dtype).itemsize
    except Exception: itemsize = 1
    need = total_px * itemsize

    # 利用可能メモリを取得 (_read_ramはMemGuardのヘルパー関数)
    # グローバルスコープに存在することを期待
    if '_read_ram' not in globals(): return True
    used, total = globals()['_read_ram']()

    avail = total - used
    # [最適化] 利用可能メモリが極端に少ない場合の安全装置を強化
    MIN_AVAIL_ESTIMATE = 512 * 1024 * 1024 # 512MB
    if avail <= MIN_AVAIL_ESTIMATE:
        # 監視が機能していない場合の推定
        if total > 0 and (used == 0 or avail <= 0):
             avail = max(MIN_AVAIL_ESTIMATE, total * 0.8)
        else:
             avail = max(avail, MIN_AVAIL_ESTIMATE)

    # 必要メモリ量が利用可能メモリの安全閾値以下か
    return need <= avail * safety

def k32_scale_integer_nn(g: Grid, ky: float, kx: float):
    # 最近傍法による整数倍スケーリング
    # [最適化] grid_to_numpyで入力を正規化(uint8)
    gg = grid_to_numpy(g)
    if ky <= 0 or kx <= 0: return grid_to_numpy([[0]])
    if ky == 1 and kx == 1: return gg
    if isinstance(np, DummyNp): return gg

    if ky >= 1 and kx >= 1:
        # 拡大
        ky_i, kx_i = int(round(ky)), int(round(kx))
        try:
            H2, W2 = gg.shape[0] * ky_i, gg.shape[1] * kx_i
        except OverflowError:
             raise MemoryError(f"K32 scale shape overflow: {ky_i}x, {kx_i}x")

        if not _mem_ok((H2, W2), gg.dtype):
            raise MemoryError(f"K32 scale OOM risk: {H2}x{W2}")

        # np.repeatによる拡大（効率的）
        return np.repeat(np.repeat(gg, ky_i, axis=0), kx_i, axis=1)

    # 縮小（間引き）
    sy = max(1, int(round(1 / ky)))
    sx = max(1, int(round(1 / kx)))
    return gg[::sy, ::sx]

def k32_tile_repeat(g: Grid, ty: int, tx: int):
    # タイリング（繰り返し）
    gg = grid_to_numpy(g)
    if ty <= 0 or tx <= 0: return grid_to_numpy([[0]])
    if ty == 1 and tx == 1: return gg
    if isinstance(np, DummyNp): return gg
    H, W = gg.shape

    try:
        H2, W2 = H * ty, W * tx
    except OverflowError:
        raise MemoryError(f"K32 tile shape overflow: {ty}x, {tx}x")

    if not _mem_ok((H2, W2), gg.dtype):
        raise MemoryError(f"K32 tile OOM risk: {H2}x{W2}")

    # [最適化] np.tileを利用して効率化しつつ、フォールバックも維持
    try:
        return np.tile(gg, (ty, tx))
    except Exception:
        # np.tile失敗時のフォールバック（手動タイリング）
        out = np.empty((H2, W2), dtype=gg.dtype)
        for r in range(ty):
            rs = r * H; re = rs + H
            for c in range(tx):
                cs = c * W; ce = cs + W
                out[rs:re, cs:ce] = gg
        return out

# ↓↓↓ KUT32 移植ブロック① (Consistency & Collatz) ↓↓↓
# ------------------------------------------------------------------------------
# Consistency & Collatz (KUT32 Patched: MDL weight is tunable)
# ------------------------------------------------------------------------------
class K32_ConsistencyChecker:
    def __init__(self, train, mdl_weight: float = 0.02, eval_fn=None):
        self.train = train
        self.mdl_weight = float(mdl_weight)
        # [最適化] k30_accuracy（最適化済み）を優先利用
        g = globals()
        self.eval_fn = eval_fn or g.get('k30_accuracy') or self._fallback_accuracy

    def set_mdl_weight(self, w: float):
        try: self.mdl_weight = max(0.0, float(w))
        except Exception: pass

    @staticmethod
    def _np_ok():
        # KUT30で定義されたグローバル変数_NP_OKを参照
        return globals().get('_NP_OK', False)

    @staticmethod
    def _fallback_accuracy(a, b):
        # k30_accuracyが利用できない場合のフォールバック
        if not K32_ConsistencyChecker._np_ok(): return 0.0
        # K30のk30_accuracyを呼び出す（依存関係を明示）
        return k30_accuracy(a, b)

    # [最適化] complexityをfloatで受け取るように変更（KUT30と整合）
    def score(self, program, complexity: float):
        accs = []
        for ain, aout in self.train:
            try:
                 # 入力もgrid_to_numpyで正規化
                ain_norm = grid_to_numpy(ain)
                accs.append(float(self.eval_fn(program(ain_norm), aout)))
            except Exception: accs.append(0.0)
        train_acc = float(sum(accs) / len(accs)) if accs else 0.0
        # ペナルティ計算をKUT30と統一
        penalty = self.mdl_weight * max(0.0, complexity - 1.0)
        return {"train_acc": train_acc, "penalty": penalty, "final": train_acc - penalty}

def k32_collatz_sequence(n: int, limit: int = 64):
    s = [n]
    while n > 1 and len(s) < limit:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s.append(n)
    return s
# ↑↑↑ KUT32 移植ブロック① 終了 ↑↑↑

# ↓↓↓ KUT32 移植ブロック② (Collatz Beam) ↓↓↓
# ------------------------------------------------------------------------------
# Collatz Beam (KUT32 Governed Beam Search, Alpha–Collatz fused)
# ------------------------------------------------------------------------------
# 注意: このビームサーチは現在のKUT32ShapeLayoutEngineからは利用されないが、基盤として準備。
# KUT30の実装と可能な限り共通化し、メンテナンス性を向上させる。
class K32_CollatzBeam:
    def __init__(self, checker, K: int,
                 monitor: ResourceMonitor = None, governor: BeamGovernor = None,
                 time_budget: float = float('inf')):
        self.checker = checker
        self.K_default = max(getattr(governor, 'min_beam', 4), int(K))
        self.monitor = monitor
        self.governor = governor
        self.time_budget = time_budget
        self.start_time = time.time()

    # --- fingerprint → (α, s0) -----------------------------------------------
    # [最適化] K30の実装と共通化
    _feat_from_train = staticmethod(K30_CollatzBeam._feat_from_train)
    _collatz_seq = staticmethod(K30_CollatzBeam._collatz_seq)

    @staticmethod
    def _alpha_from_feat(colors, components, H, W, symmetry, base=1.0):
        # KUT32用パラメータ (移植コード維持)
        chi = 0.4*math.log1p(colors) + 0.4*math.log1p(components) + 0.2*math.log1p(H*W) - 0.2*symmetry
        a = 0.7 + 0.5*chi
        return max(0.7, min(1.5, base * a / 1.2))

    @staticmethod
    def _s0_from_feat(colors, components, H, W, symmetry, scale=0.75):
        # KUT32用パラメータ (移植コード維持)
        chi = 0.4*math.log1p(colors) + 0.4*math.log1p(components) + 0.2*math.log1p(H*W) - 0.2*symmetry
        s0 = int(round(40*chi) + 5)
        s0 = max(7, min(63, int(round(s0 * scale))))
        if s0 % 2 == 0: s0 += 1
        return s0

    @staticmethod
    def _phase_time_fracs(n:int):
        # (移植コード維持)
        base = [0.30,0.25,0.20,0.12,0.08,0.05,0.04,0.03][:max(1, n)]
        s = sum(base[:n]);  return [x/s for x in base[:n]]

    @staticmethod
    def _fused_mult(alpha_eff: float, phase: str):
        # (移植コード維持)
        K_mult = (alpha_eff ** 1.20) * (1.15 if phase == 'odd' else 0.75)
        MDL_mult = (1.0 / alpha_eff) * (0.88 if phase == 'odd' else 1.20)
        return K_mult, MDL_mult

    def derive_budget(self):
        # [最適化] k30_hammingを利用し、範囲を7-35に調整
        errs = [k30_hamming(ain,aout) for ain,aout in self.checker.train]
        if not errs: return 7
        E0 = int(statistics.mean(errs))
        return max(7, min(35, E0 // 8 + 7))

    def search(self, base_ops, masks_fn, train):
        # (移植コード維持。Governor連携ログの微調整とcomplexityの型整合)
        if not train: return (lambda g: g), {"final": 0.0}

        # α・s0 決定
        colors,components,H,W,sym = self._feat_from_train(train)
        alpha0 = self._alpha_from_feat(colors,components,H,W,sym,base=1.0)
        s0     = self._s0_from_feat(colors,components,H,W,sym,scale=0.75)
        seq    = self._collatz_seq(s0, max_steps=8)
        # fracs  = self._phase_time_fracs(len(seq)) # 未使用

        # 初期クラスタ数
        masks0 = masks_fn(train[0][0]) if callable(masks_fn) else []
        n_clusters = len(masks0) if masks0 else 1
        def _mk_empty(C): return [[] for _ in range(C)]
        def _prog_from(mf, seqs):
            def _p(g):
                # 入力gはgrid_to_numpyで正規化済みと仮定
                masks = mf(g)
                if not masks: return g.copy()
                C = min(len(masks), len(seqs)); preds=[]
                for m, sops in zip(masks[:C], seqs[:C]):
                    sub = g.copy()
                    for op in sops:
                        try: sub = op.apply(sub, m)
                        except Exception: pass
                    preds.append(sub)
                # overlay (k30のoverlayを利用)
                try:
                    return overlay(g, masks[:C], preds)
                except Exception:
                    return g.copy()
            return _p

        # [最適化] complexityをfloatに
        prog0 = _prog_from(masks_fn, _mk_empty(n_clusters))
        sc0 = self.checker.score(prog0, 0.0)["final"]
        beam = [(_mk_empty(n_clusters), sc0, 0.0)]

        N0 = self.derive_budget()
        lengths = [1] + [max(1, n % 5 + 1) for n in k32_collatz_sequence(N0)]
        current_K = int(self.K_default)
        base_mdl = float(getattr(self.checker, "mdl_weight", 0.02))

        for step_idx, L in enumerate(lengths[:32]): # 最大ステップ数増加
            phase, sval = seq[min(step_idx, len(seq)-1)]
            alpha_eff = max(0.6, min(1.6, alpha0 * (1.18 if phase == 'odd' else 0.88)))
            K_mult, MDL_mult = self._fused_mult(alpha_eff, phase)
            phase_K = max(4, int(round(self.K_default * K_mult)))
            self.checker.set_mdl_weight(base_mdl * MDL_mult)

            # Governor 連携 (KUT30とロジック統一)
            if self.governor and self.monitor:
                try:
                    snap = self.monitor.snapshot()
                    elapsed = time.time() - self.start_time
                    default_K = min(phase_K, self.governor.max_beam)
                    k_suggest, _, reason = self.governor.suggest(snap, default_K, default_K, elapsed, self.time_budget)
                    if isinstance(k_suggest, int) and k_suggest > 0:
                        if k_suggest != current_K:
                            ramf = float(getattr(snap,"ram_frac")() if hasattr(snap,"ram_frac") else 0.0)
                            vramfs = snap.vram_fracs(); vramf = max(vramfs) if vramfs else 0.0
                            # [最適化] ログ出力形式をKUT30と統一
                            print(f"[KUT32-Gov] {time.strftime('%H:%M:%S')} | Phase:{phase}(α={alpha_eff:.2f}) | Time: {elapsed:.1f}s/{self.time_budget:.0f}s | Beam: {current_K} -> {k_suggest} (Target:{phase_K}) | Reason: {reason} | RAM: {ramf*100:.1f}% VRAM: {vramf*100:.1f}%", file=sys.stderr)
                        current_K = max(self.governor.min_beam, min(self.governor.max_beam, k_suggest))
                    else:
                        current_K = phase_K

                    ramf_check = float(getattr(snap,"ram_frac")() if hasattr(snap,"ram_frac") else 0.0)
                    # Governorのハードリミット設定（デフォルト0.88）を参照
                    HARD_RAM_LIMIT = getattr(self.governor,"hard_ram", 0.88)
                    if ramf_check > HARD_RAM_LIMIT:
                        current_K = max(self.governor.min_beam, max(4, current_K // 2)); gc.collect()
                        print(f"[KUT32-Gov] EMERGENCY: High RAM usage ({ramf_check*100:.1f}% > {HARD_RAM_LIMIT*100:.1f}%). Beam reduced to {current_K}. GC called.", file=sys.stderr)

                    if len(beam) > current_K: beam = beam[:current_K]
                except Exception as e:
                    print(f"[KUT32-Gov] Governor error: {e}", file=sys.stderr)
                    current_K = phase_K
            else:
                current_K = phase_K

            # 拡張
            newbeam = []
            for seqs, sc, comp in beam:
                for cidx in range(n_clusters):
                    for op in base_ops:
                        nseqs = [s[:] for s in seqs]
                        nseqs[cidx] = nseqs[cidx] + [op]
                        prog = _prog_from(masks_fn, nseqs)
                        # [最適化] complexity(float)を加算
                        new_comp = comp + float(getattr(op, 'complexity', 1.0))
                        score = self.checker.score(prog, new_comp)
                        newbeam.append((nseqs, score["final"], new_comp))
            if not newbeam: break

            # K 上位
            if len(newbeam) > current_K:
                beam = heapq.nlargest(current_K, newbeam, key=lambda t: t[1])
            else:
                newbeam.sort(key=lambda t: t[1], reverse=True)
                beam = newbeam

        if not beam: return (lambda g: grid_to_numpy(g)), {"final": 0.0}
        best_seqs, best_score, best_comp = max(beam, key=lambda t: t[1])
        best_prog = _prog_from(masks_fn, best_seqs)
        metrics = self.checker.score(best_prog, best_comp)
        metrics["final"] = float(best_score)
        return best_prog, metrics
# ↑↑↑ KUT32 移植ブロック② 終了 ↑↑↑

# ---- KUT32 Engine ----
K32_Pair = Tuple[Grid, Grid]
class KUT32ShapeLayoutEngine:
    def __init__(self):
        self.best_transform: Optional[Callable[[Any], Grid]] = None
        self.metrics: dict = {"accuracy": 0.0}
        self.transform_name: str = "Identity"

    def fit(self, train: List[K32_Pair]):
        if not train: return self

        # 形状比率の計算と一貫性のチェック
        shape_ratios = []
        # consistent_shape = True # ループ内で管理不要

        for ain_raw, aout_raw in train:
            # [最適化] grid_to_numpyで正規化
            ain = grid_to_numpy(ain_raw); aout = grid_to_numpy(aout_raw)
            shape_in = getattr(ain, 'shape', (0,0))
            shape_out = getattr(aout, 'shape', (0,0))

            # ゼロ除算防止チェック
            if shape_in[0] == 0 or shape_in[1] == 0: continue

            try:
                ry, rx = shape_out[0] / shape_in[0], shape_out[1] / shape_in[1]
                shape_ratios.append((ry, rx))
            except ZeroDivisionError:
                continue

        # 一貫性のチェック
        if not shape_ratios or len(set(shape_ratios)) > 1:
            return self

        ry, rx = shape_ratios[0]
        candidates: List[Tuple[str, Callable[[Any], Grid]]] = []

        # 安全性チェック関数
        def _gate_all_pairs_is_safe(kind: str) -> bool:
            for ain_raw, aout_raw in train:
                ain = grid_to_numpy(ain_raw); aout = grid_to_numpy(aout_raw)
                shape_in = getattr(ain, 'shape', (0,0))
                shape_out = getattr(aout, 'shape', (0,0))

                if kind == "scale":
                    H2 = int(round(shape_in[0] * ry)); W2 = int(round(shape_in[1] * rx))
                elif kind == "tile" and float(ry).is_integer() and float(rx).is_integer():
                    ty, tx = int(ry), int(rx)
                    try:
                        H2, W2 = shape_in[0] * ty, shape_in[1] * tx
                    except OverflowError: return False
                else:
                    return False

                # 予測される形状と実際の出力形状が一致し、かつメモリ安全か確認
                if (H2, W2) != shape_out: return False
                # dtypeも渡してチェック
                if not _mem_ok((H2, W2), getattr(ain, 'dtype', None)): return False
            return True

        # 候補生成
        # 1. スケーリング
        if _gate_all_pairs_is_safe("scale"):
            def scale_fn(g: Any, k_y=ry, k_x=rx) -> Grid:
                # 入力はk32_scale_integer_nn内でgrid_to_numpyされる
                return k32_scale_integer_nn(g, k_y, k_x)
            candidates.append(("ScaleNN", scale_fn))

        # 2. タイリング（比率が整数の場合のみ）
        if float(ry).is_integer() and float(rx).is_integer():
            if _gate_all_pairs_is_safe("tile"):
                ty, tx = int(ry), int(rx)
                def tile_fn(g: Any, t_y=ty, t_x=tx) -> Grid:
                    return k32_tile_repeat(g, t_y, t_x)
                candidates.append(("TileRepeat", tile_fn))

        if not candidates: return self

        # 評価（学習データに対する精度）
        best_score = -1.0
        for name, transform in candidates:
            accs = []
            for ain_raw, aout_raw in train:
                ain = grid_to_numpy(ain_raw); aout = grid_to_numpy(aout_raw)
                try:
                    pred = transform(ain)
                    # [最適化] k30_accuracyを利用
                    accs.append(k30_accuracy(pred, aout))
                except MemoryError as e:
                    print(f"[KUT32] MemoryError during evaluation: {e}", file=sys.stderr)
                    accs.append(0.0)
                except Exception:
                    accs.append(0.0)

            score = float(statistics.mean(accs)) if accs else 0.0
            if score > best_score:
                best_score = score
                self.best_transform = transform
                self.transform_name = name
                self.metrics["accuracy"] = score
        return self

def make_kut32_candidate_from_arctask(task: ARCTask) -> Tuple[str, float, Callable[[Grid], Grid]]:
    train_pairs: List[K32_Pair] = []
    try:
        # ARCTaskのデータはすでにNumPy配列化されている
        for p in getattr(task, "train_pairs", []):
            ain = p.get("input")
            aout = p.get("output")
            if ain is not None and aout is not None:
                train_pairs.append((ain, aout))
    except Exception:
        pass

    engine = KUT32ShapeLayoutEngine().fit(train_pairs)
    transform = engine.best_transform
    acc = float(engine.metrics.get("accuracy", 0.0))

    if transform is None or acc < 0.5:
        # [最適化] 失敗時のフォールバックもgrid_to_numpyで安全に処理
        return "KUT32-ShapeLayout[None]", 0.0, lambda g: grid_to_numpy(g)

    # 信頼度の計算（0.65から0.99の範囲）。形状一致は信頼度高め。
    conf = max(0.65, min(0.99, acc * 0.98))
    source = f"KUT32-ShapeLayout[{engine.transform_name}]"

    # 入力変換(grid_to_numpy)を含むラッパー関数を返す
    return source, conf, lambda g, t=transform: t(grid_to_numpy(g))

# ==============================================================================
# 8. kutos/engines_other.py (ALO, GeomMatch)
#    高速な幾何学変換とオブジェクトマッチングエンジン
# ==============================================================================
# (OMUX4o: Alpha-Collatz Fusion Beam 移植済 + ヘルパー整理・GeomMatch堅牢化・Governor連携強化)

# [最適化] ローカル安全ヘルパを整理・削除。common.pyやk30の関数(grid_to_numpy, k30_accuracy)を利用する。
# _is_np_array -> 削除
# _to_np -> 削除
# _avg_accuracy -> 削除

# D4 変換（内部定義）
def _d4_table():
    if isinstance(np, DummyNp):
        return {'I': lambda m: m}
    try:
        # 基本的なD4変換セット
        return {
            'I':    lambda m: m,
            'R90':  lambda m: np.rot90(m, 1),
            'R180': lambda m: np.rot90(m, 2),
            'R270': lambda m: np.rot90(m, 3),
            'FH':   lambda m: np.fliplr(m), # 水平反転
            'FV':   lambda m: np.flipud(m), # 垂直反転
        }
    except Exception:
        return {'I': lambda m: m}

# --- Engine 5: ALO (Aggressive Logic Ops) ---
class ALOHeuristicEngine:
    # 基本的なD4幾何学変換を高速にテスト
    def __init__(self, am: AbstractionModule):
        self.abstraction = am
        self.transforms = _d4_table()

    def _score_transform(self, name: str, fn, train_pairs: List[Dict]) -> float:
        # 学習ペアに対する変換の適合度（平均精度）を計算
        if not train_pairs:
            return 0.5 if name == 'I' else 0.4

        accs = []
        for p in train_pairs:
            try:
                # input/outputはARCTaskでNumPy化(uint8)済み
                ain, aout = p['input'], p['output']
                # 変換を適用
                pred = fn(ain)
                # [最適化] k30_accuracyを利用
                accs.append(k30_accuracy(pred, aout))
            except Exception:
                accs.append(0.0)
        return sum(accs)/len(accs) if accs else 0.0

    def solve(self, task: ARCTask, budget: float, safe_mode: bool) -> List[CandidateSolution]:
        cands: List[CandidateSolution] = []
        scores = []

        # 全てのD4変換を評価
        for name, fn in self.transforms.items():
            try:
                s = self._score_transform(name, fn, getattr(task, "train_pairs", []))
                scores.append((s, name, fn))
            except Exception:
                continue

        if not scores: return []
        scores.sort(reverse=True, key=lambda t: t[0])

        # 上位N個を採用（AutoTune設定を参照）
        MAX_CAND = CONFIG.AUTOTUNE_CFG.get("MAX_CAND_PER_OP", 3)
        top = scores[:MAX_CAND]

        for s, name, fn in top:
            # [最適化] 信頼度の計算範囲を微調整（0.55から0.95の範囲）
            conf = max(0.55, min(0.95, 0.88 * float(s) + 0.07))

            def _solver(g, f=fn):
                try:
                    # 入力gはCandidateSolution.solve経由で渡されるため、grid_to_numpyで安全確保
                    return f(grid_to_numpy(g))
                except Exception:
                    # 失敗時は入力を安全に返す
                    return grid_to_numpy(g)

            cands.append(CandidateSolution(source=f"ALO[{name}]",
                                           confidence=float(conf),
                                           solver_logic=_solver))
        return cands

# ↓↓↓ Geom 移植ブロック① (Consistency & Collatz) ↓↓↓
# ------------------------------------------------------------------------------
# Consistency & Collatz (Geom Patched)
# ------------------------------------------------------------------------------
# (KUT30/32と同様の最適化を適用: k30_accuracy優先利用、complexity float化)
class GEOM_ConsistencyChecker:
    def __init__(self, train, mdl_weight: float = 0.02, eval_fn=None):
        self.train = train
        self.mdl_weight = float(mdl_weight)
        g = globals()
        # [最適化] k30_accuracyを優先利用
        self.eval_fn = eval_fn or g.get('k30_accuracy') or self._fallback_accuracy

    def set_mdl_weight(self, w: float):
        try: self.mdl_weight = max(0.0, float(w))
        except Exception: pass

    @staticmethod
    def _fallback_accuracy(a, b):
        # k30_accuracyが利用できない場合のフォールバック
        return k30_accuracy(a, b)

    # [最適化] complexityをfloatで受け取る
    def score(self, program, complexity: float):
        accs = []
        for ain, aout in self.train:
            try:
                # 入力もgrid_to_numpyで正規化
                ain_norm = grid_to_numpy(ain)
                accs.append(float(self.eval_fn(program(ain_norm), aout)))
            except Exception: accs.append(0.0)
        train_acc = float(sum(accs)/len(accs)) if accs else 0.0
        # ペナルティ計算をKUT30と統一
        penalty = self.mdl_weight * max(0.0, complexity - 1.0)
        return {"train_acc": train_acc, "penalty": penalty, "final": train_acc - penalty}

def geom_collatz_sequence(n: int, limit: int = 64):
    s=[n]
    while n>1 and len(s)<limit:
        n = n//2 if n%2==0 else 3*n+1
        s.append(n)
    return s
# ↑↑↑ Geom 移植ブロック① 終了 ↑↑↑

# ↓↓↓ Geom 移植ブロック② (Collatz Beam) ↓↓↓
# ------------------------------------------------------------------------------
# Collatz Beam (Geom Governed Beam Search, Alpha–Collatz fused)
# ------------------------------------------------------------------------------
# 注意: Geomのビームサーチは、現在のKUTGeomMatchEngineからは利用されない。
class GEOM_CollatzBeam:
    # (v8.2: KUT30との共通化、Governor連携最適化、探索予算拡大)
    def __init__(self, checker, K: int,
                 monitor: ResourceMonitor = None, governor: BeamGovernor = None,
                 time_budget: float = float('inf')):
        self.checker = checker
        # [最適化] Governorの最小ビーム幅設定との整合性を確保
        self.K_default = max(getattr(governor, 'min_beam', 4), int(K))
        self.monitor = monitor
        self.governor = governor
        self.time_budget = time_budget
        self.start_time = time.time()

    # [最適化] K30と実装を共通化 (K30_CollatzBeamが定義済みであることを前提とする)
    # グローバルスコープからK30_CollatzBeamを参照
    try:
        _K30_CB = globals().get('K30_CollatzBeam')
        if _K30_CB:
            # staticmethodでラップして自クラスのメソッドとして登録
            _feat_from_train = staticmethod(_K30_CB._feat_from_train)
            _collatz_seq = staticmethod(_K30_CB._collatz_seq)
        else:
             raise NameError("K30_CollatzBeam not found for common logic reuse.")
    except Exception as e:
        # K30が利用できない場合のフォールバック（移植コードのロジックを維持）
        print(f"[GEOM Beam] Warning: Failed to reuse K30 logic: {e}. Using fallback definitions.", file=sys.stderr)
        @staticmethod
        def _feat_from_train(train):
            # (移植コードのフォールバック実装)
            try:
                g = train[0][0]
                H, W = int(getattr(g,"shape",(0,0))[0]), int(getattr(g,"shape",(0,0))[1])
            except Exception: return (6,8,20,20,0.3)
            try: colors = int(np.unique(g).size)
            except Exception: colors = 6
            components = 10 # 簡易推定
            try: symmetry = float((g == g[:, ::-1]).mean())
            except Exception: symmetry = 0.3
            return (max(1,colors), max(1,components), max(1,H), max(1,W), symmetry)

        @staticmethod
        def _collatz_seq(s0:int, max_steps:int=8):
            # (移植コードのフォールバック実装)
            s, seq = s0, []
            for _ in range(max_steps):
                phase = 'odd' if (s % 2 == 1) else 'even'
                seq.append((phase, s))
                if s == 1: break
                s = 3*s + 1 if phase == 'odd' else s // 2
            return seq


    @staticmethod
    def _alpha_from_feat(colors,components,H,W,symmetry,base=1.0):
        # (移植コード維持) Geom用パラメータ
        chi = 0.45*math.log1p(colors) + 0.45*math.log1p(components) + 0.10*math.log1p(H*W) - 0.2*symmetry
        a = 0.7 + 0.5*chi
        return max(0.7, min(1.6, base * a / 1.15))

    @staticmethod
    def _s0_from_feat(colors,components,H,W,symmetry,scale=0.8):
        # (移植コード維持) Geom用パラメータ
        chi = 0.45*math.log1p(colors) + 0.45*math.log1p(components) + 0.10*math.log1p(H*W) - 0.2*symmetry
        s0 = int(round(40*chi) + 5)
        s0 = max(7, min(63, int(round(s0 * scale))))
        if s0 % 2 == 0: s0 += 1
        return s0

    @staticmethod
    def _phase_time_fracs(n:int):
        # (移植コード維持)
        base=[0.30,0.25,0.20,0.12,0.08,0.05,0.04,0.03][:max(1,n)]
        s=sum(base[:n]); return [x/s for x in base[:n]]

    @staticmethod
    def _fused_mult(alpha_eff: float, phase: str):
        # (移植コード維持)
        K_mult  = (alpha_eff ** 1.10) * (1.12 if phase=='odd' else 0.78)
        MDL_mult = (1.0 / alpha_eff) * (0.90 if phase=='odd' else 1.25)
        return K_mult, MDL_mult

    def derive_budget(self):
        # [最適化] k30_hammingを利用し、範囲を7-35に調整 (KUT30/32と統一)
        # k30_hammingはグローバルスコープで利用可能
        errs = [k30_hamming(ain,aout) for ain,aout in self.checker.train]
        if not errs: return 7
        E0=int(statistics.mean(errs))
        return max(7, min(35, E0//8 + 7))

    def search(self, base_ops, masks_fn, train):
        # (Governor連携強化とcomplexityの型整合)
        if not train: return (lambda g:g), {"final":0.0}

        colors,components,H,W,sym = self._feat_from_train(train)
        alpha0 = self._alpha_from_feat(colors,components,H,W,sym,base=1.0)
        s0     = self._s0_from_feat(colors,components,H,W,sym,scale=0.8)
        seq    = self._collatz_seq(s0, max_steps=8)
        # fracs  = self._phase_time_fracs(len(seq)) # 未使用

        masks0 = masks_fn(train[0][0]) if callable(masks_fn) else []
        n_clusters = len(masks0) if masks0 else 1
        def _mk_empty(C): return [[] for _ in range(C)]
        def _prog_from(mf, seqs):
            def _p(g):
                # 入力gはgrid_to_numpyで正規化済みと仮定
                masks = mf(g)
                if not masks: return g.copy()
                C=min(len(masks), len(seqs)); preds=[]
                for m,sops in zip(masks[:C], seqs[:C]):
                    sub=g.copy()
                    for op in sops:
                        try: sub=op.apply(sub,m)
                        except Exception: pass
                    preds.append(sub)
                try:
                    # k30のoverlayを利用
                    return overlay(g, masks[:C], preds)
                except Exception:
                    return g.copy()
            return _p

        # [最適化] complexityをfloatに
        prog0 = _prog_from(masks_fn, _mk_empty(n_clusters))
        sc0 = self.checker.score(prog0, 0.0)["final"]
        beam=[(_mk_empty(n_clusters), sc0, 0.0)]

        N0=self.derive_budget()
        lengths=[1]+[max(1, n%5+1) for n in geom_collatz_sequence(N0)]
        current_K=int(self.K_default)
        base_mdl=float(getattr(self.checker,"mdl_weight",0.02))

        # [最適化] 最大ステップ数増加 (24 -> 32)
        for step_idx, L in enumerate(lengths[:32]):
            phase, sval = seq[min(step_idx, len(seq)-1)]
            alpha_eff = max(0.6, min(1.6, alpha0 * (1.18 if phase=='odd' else 0.88)))
            K_mult, MDL_mult = self._fused_mult(alpha_eff, phase)
            phase_K = max(4, int(round(self.K_default * K_mult)))
            self.checker.set_mdl_weight(base_mdl * MDL_mult)

            # [最適化] Governor連携 (KUT30/32とロジック統一・強化)
            if self.governor and self.monitor:
                try:
                    snap=self.monitor.snapshot()
                    elapsed=time.time()-self.start_time
                    # Governorの最大ビーム幅（例: 128）を考慮
                    default_K=min(phase_K, self.governor.max_beam)
                    k_suggest,_,reason=self.governor.suggest(snap, default_K, default_K, elapsed, self.time_budget)

                    if isinstance(k_suggest,int) and k_suggest>0:
                        if k_suggest!=current_K:
                            ramf=float(getattr(snap,"ram_frac")() if hasattr(snap,"ram_frac") else 0.0)
                            vramfs=snap.vram_fracs(); vramf=max(vramfs) if vramfs else 0.0
                            # ログ出力形式をKUT30と統一
                            print(f"[GEOM-Gov] {time.strftime('%H:%M:%S')} | Phase:{phase}(α={alpha_eff:.2f}) | Time: {elapsed:.1f}s/{self.time_budget:.0f}s | Beam: {current_K} -> {k_suggest} (Target:{phase_K}) | Reason: {reason} | RAM: {ramf*100:.1f}% VRAM: {vramf*100:.1f}%", file=sys.stderr)
                        current_K=max(self.governor.min_beam, min(self.governor.max_beam, k_suggest))
                    else:
                        current_K=phase_K

                    # 緊急対応ロジック強化
                    ramf_check=float(getattr(snap,"ram_frac")() if hasattr(snap,"ram_frac") else 0.0)
                    # [最適化] Governorのハードリミット設定（最適化されたデフォルト0.88）を参照。ハードコードされた0.90を撤廃。
                    HARD_RAM_LIMIT = getattr(self.governor,"hard_ram", 0.88)
                    if ramf_check > HARD_RAM_LIMIT:
                        current_K=max(self.governor.min_beam, max(4, current_K//2)); gc.collect()
                        print(f"[GEOM-Gov] EMERGENCY: High RAM usage ({ramf_check*100:.1f}% > {HARD_RAM_LIMIT*100:.1f}%). Beam reduced to {current_K}. GC called.", file=sys.stderr)

                    if len(beam)>current_K: beam=beam[:current_K]
                except Exception as e:
                    print(f"[GEOM-Gov] Governor error: {e}", file=sys.stderr)
                    current_K=phase_K
            else:
                current_K=phase_K

            newbeam=[]
            for seqs, sc, comp in beam:
                for cidx in range(n_clusters):
                    for op in base_ops:
                        nseqs=[s[:] for s in seqs]
                        nseqs[cidx]=nseqs[cidx]+[op]
                        prog=_prog_from(masks_fn, nseqs)
                        # [最適化] complexity(float)を加算
                        new_comp = comp + float(getattr(op, 'complexity', 1.0))
                        score=self.checker.score(prog, new_comp)
                        newbeam.append((nseqs, score["final"], new_comp))
            if not newbeam: break

            if len(newbeam)>current_K:
                beam=heapq.nlargest(current_K, newbeam, key=lambda t:t[1])
            else:
                newbeam.sort(key=lambda t:t[1], reverse=True)
                beam=newbeam

        # [最適化] 失敗時のフォールバックもgrid_to_numpyで安全に処理
        if not beam: return (lambda g: grid_to_numpy(g)), {"final":0.0}
        best_seqs, best_score, best_comp=max(beam, key=lambda t:t[1])
        best_prog=_prog_from(masks_fn, best_seqs)
        metrics=self.checker.score(best_prog, best_comp)
        metrics["final"]=float(best_score)
        return best_prog, metrics
# ↑↑↑ Geom 移植ブロック② 終了 ↑↑↑

# --- Engine 6: GeomMatch (Geometric Alignment) ---
class KUTGeomMatchEngine:
    # 主要オブジェクト間の幾何学的位置合わせ（D4+平行移動）を検出
    def __init__(self, am: AbstractionModule):
        self.abstraction = am
        self._d4 = _d4_table()

    def _largest_object(self, scene: "SceneGraph"):
        # シーン内の最大のオブジェクトを取得
        # [最適化] SceneObject.area属性を利用して効率化 (core_modules.pyで最適化済み前提)
        try:
            if not hasattr(scene, "objects") or not scene.objects: return None
            # area属性で最大オブジェクトを選択
            return max(scene.objects, key=lambda o: getattr(o, 'area', 0))
        except Exception:
            # フォールバック（従来のピクセル数カウント）
            try:
                best, best_n = None, -1
                for o in scene.objects:
                    n = int(getattr(getattr(o, "pixels", None), "shape", [0])[0]) if hasattr(o, "pixels") else 0
                    if n > best_n: best, best_n = o, n
                return best
            except Exception:
                return None

    def _compose(self, tname: str, dy: int, dx: int):
        # 変換（D4+平行移動）を実行するソルバー関数を生成
        # [最適化] 背景色の確実な取得と、安全な平行移動ロジック(スライシング)の導入
        def _solver(grid: Any, _t=tname, _dy=dy, _dx=dx):
            # 入力はgrid_to_numpyで正規化(uint8)
            g = grid_to_numpy(grid)
            try:
                # 1. D4変換
                tf = self._d4.get(_t, lambda x: x)
                gT = tf(g)

                # 2. 背景色の決定（抽象化モジュールを利用して精度向上）
                try:
                    # 変換後のグリッドから背景色を取得
                    sceneT = self.abstraction.abstract(gT)
                    bg = sceneT.background_color
                except Exception:
                    # 失敗時はk30_mode_colorで代用（グローバルスコープに存在前提）
                    if 'k30_mode_color' in globals():
                        bg = k30_mode_color(gT)
                    else:
                        # 最終手段
                        try:
                            vals, cnts = np.unique(gT, return_counts=True)
                            bg = int(vals[cnts.argmax()]) if vals.size else 0
                        except: bg = 0

                # 3. 平行移動 (安全なシフトロジック)
                H, W = gT.shape
                # 背景色で初期化 (dtypeはgT=uint8を維持)
                out = np.full_like(gT, bg)

                # 安全なコピー範囲を計算
                # ソース範囲
                r_src_start, r_src_end = max(0, -_dy), min(H, H - _dy)
                c_src_start, c_src_end = max(0, -_dx), min(W, W - _dx)
                # デスティネーション範囲
                r_dst_start, r_dst_end = max(0, _dy), min(H, H + _dy)
                c_dst_start, c_dst_end = max(0, _dx), min(W, W + _dx)

                # 有効な範囲がある場合のみコピー
                if r_src_start < r_src_end and c_src_start < c_src_end:
                    out[r_dst_start:r_dst_end, c_dst_start:c_dst_end] = gT[r_src_start:r_src_end, c_src_start:c_src_end]

                return out
            except Exception:
                # 失敗時は入力を返す
                return g
        return _solver

    def solve(self, task: ARCTask, budget: float, safe_mode: bool) -> List[CandidateSolution]:
        # 依存チェック：SciPyが無ければスキップ
        if scipy_label is None or isinstance(np, DummyNp):
            return []

        votes = []

        # AutoTune設定を参照
        GEOM_TRIALS = CONFIG.AUTOTUNE_CFG.get("GEOM_TRIALS", 4) # デフォルト値を引き上げ
        # [最適化] CONFIGから最適化済みのMAX_PAIRSを取得 (config.pyで最適化済み前提)
        MAX_PAIRS = getattr(CONFIG, "GEOM_MATCH_MAX_PAIRS", 300)

        # 学習ペアごとにアラインメントを計算し、投票
        for _ in range(GEOM_TRIALS): # 試行回数分繰り返す
            for pair in getattr(task, "train_pairs", []):
                try:
                    # 入出力はNumPy化済み
                    in_scene = self.abstraction.abstract(pair['input'])
                    out_scene = self.abstraction.abstract(pair['output'])

                    # 最大オブジェクトを取得
                    A = self._largest_object(in_scene)
                    B = self._largest_object(out_scene)
                    if A is None or B is None: continue

                    # アラインメント計算 (探索ペア数を渡す)
                    # align_componentsはgeometry_utils.pyで定義
                    res = align_components(A, B, max_pairs=MAX_PAIRS)
                    tname = res.get('transform') or 'I'
                    dy, dx = int(res.get('dy', 0)), int(res.get('dx', 0))
                    iou = float(res.get('iou', 0.0))

                    # [最適化] 閾値を調整 (0.50 -> 0.60) 精度向上
                    if iou > 0.60:
                        votes.append((tname, dy, dx, iou))
                except Exception:
                    continue

        if not votes: return []

        # 投票結果の集計（変換タイプごとに中央値を採用）
        agg: Dict[str, Dict[str, Any]] = {}
        for t, dy, dx, iou in votes:
            d = agg.setdefault(t, {"dys": [], "dxs": [], "ious": []})
            d["dys"].append(dy); d["dxs"].append(dx); d["ious"].append(iou)

        ranked = []
        for t, d in agg.items():
            try:
                # [最適化] 安全性強化: 空リストチェック
                if not d["dys"] or not d["dxs"] or not d["ious"]: continue

                med_dy = int(round(float(np.median(d["dys"]))))
                med_dx = int(round(float(np.median(d["dxs"]))))
                mean_iou = float(np.mean(d["ious"]))
                support = len(d["ious"])
                # スコア計算（IoUとサポート数を考慮）
                score = mean_iou * (1.0 + 0.05 * support)
                ranked.append((score, t, med_dy, med_dx, mean_iou, support))
            except Exception:
                continue

        if not ranked: return []

        ranked.sort(reverse=True, key=lambda x: x[0])

        # 上位2つを採用
        top = ranked[:2]
        cands: List[CandidateSolution] = []
        for score, tname, dy, dx, mean_iou, support in top:
            # [最適化] 信頼度の計算範囲を調整（0.60から0.95の範囲）
            conf = max(0.60, min(0.95, 0.65 + 0.35 * mean_iou + 0.02 * min(10, support)))
            cands.append(CandidateSolution(
                source=f"GeomMatch[{tname},dy={dy},dx={dx}]",
                confidence=float(conf),
                solver_logic=self._compose(tname, dy, dx)
            ))
        return cands
        
# ==============================================================================
# 9. kutos/pipeline.py (Pipeline and Arbitrator)
#    全体のパイプライン制御と最終的な解の選択
# ==============================================================================
# (OMUX4o: Adaptive Budget Allocation, 実行順序最適化, Governor最適設定適用, 堅牢性強化)

class ArbitratorModule:
    # 複数の候補から最終的な提出（2つ）を選択
    def __init__(self, cc: ConsistencyChecker):
        self.checker = cc

    def _are_grids_equal(self, g1: Grid, g2: Grid) -> bool:
        # [最適化] 堅牢なグリッド比較ヘルパー
        shape1 = getattr(g1, 'shape', None)
        shape2 = getattr(g2, 'shape', None)

        # NumPy環境での比較
        if not isinstance(np, DummyNp) and isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray):
            if shape1 != shape2: return False
            try:
                # 内容比較
                return np.array_equal(g1, g2)
            except Exception:
                return False # 比較失敗時は異なるとみなす

        # リスト環境またはフォールバック (numpy_to_listで安全に変換して比較)
        l1 = numpy_to_list(g1)
        l2 = numpy_to_list(g2)
        return l1 == l2

    def select_final_submissions(self, cands: List[CandidateSolution], t_in: Grid) -> Tuple[Grid, Grid]:
        # 入力(t_in)はARCTaskでNumPy化済み

        # [最適化] フォールバック解を安全なコピーで確保
        fallback_solution = grid_to_numpy(t_in)

        if not cands: return fallback_solution, fallback_solution

        s_cands = []
        # 各候補を評価（信頼度と幾何学的一貫性を組み合わせる）
        for c in cands:
            # CandidateSolution.solveは内部で安全性を確保し、失敗時はNoneを返す
            out = c.solve(t_in)
            if out is not None:
                # ConsistencyChecker.evaluateで一貫性スコア(0.0-1.0)を計算
                consistency_score = self.checker.evaluate(c, t_in)
                # 最終スコア = 信頼度 * 一貫性スコア (両方が高いことを重視)
                score = c.confidence * consistency_score
                s_cands.append({'output': out, 'score': score, 'source': c.source, 'conf': c.confidence, 'consist': consistency_score})

        if not s_cands: return fallback_solution, fallback_solution

        s_cands.sort(key=lambda x: x['score'], reverse=True)

        # 計算得点のログ出力 (Top 6、詳細表示)
        score_log = " | ".join([f"[{c['source']}] Score:{c['score']:.4f} (Conf:{c['conf']:.2f}*Consist:{c['consist']:.2f})" for c in s_cands[:6]])
        print(f"[Arbitrator] Scores: {score_log}", file=sys.stderr)

        # 1番目の解
        a1 = s_cands[0]['output']
        a2 = a1 # デフォルトでは1番目と同じ

        # 2番目の解（1番目と異なるもの）
        for c in s_cands[1:]:
            if not self._are_grids_equal(a1, c['output']):
                a2 = c['output']
                break

        # 常に2つの解を返す（内部形式用、NumPy配列）
        return a1, a2

class KUT_OS_Solver_Internal:
    # ソルバー本体（全エンジンの統合）
    def __init__(self):
        self.am = AbstractionModule(); self.af = AugmentationFramework()

        # 6基のエンジン初期化
        self.ie = KUTInductionEngine(self.am)
        self.te = KUTTransductionEngine(self.am)
        self.gm = KUTGeomMatchEngine(self.am)
        self.alo = ALOHeuristicEngine(self.am)
        self.make_kut30_candidate = make_kut30_candidate_from_arctask
        self.make_kut32_candidate = make_kut32_candidate_from_arctask

        self.cc = ConsistencyChecker(self.af); self.arb = ArbitratorModule(self.cc)

        # MemGuard (Monitor & Governor) 初期化
        self.monitor = None; self.governor = None
        self._initialize_memguard()

    def _initialize_memguard(self):
        # [最適化] MemGuardの初期化と最適設定の適用
        try:
            monitor_disk_path = "/kaggle/working" if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else "./"

            # 1. Monitor起動
            if 'threading' in globals() and hasattr(threading, 'Thread') and threading.Thread is not None:
                # サンプリング間隔を1.5秒に調整（精度と負荷のバランス）
                self.monitor = ResourceMonitor(sample_sec=1.5, disk_path=monitor_disk_path).start()
            else:
                 print(f"[Pipeline] INFO: Threading disabled. ResourceMonitor will not run in background.", file=sys.stderr)

            # 2. Governor設定 (ARC 2025 12H Optimized Parameters)
            # MemGuard最適化で設定された積極的なパラメータを明示的に適用。
            # 最大ビーム幅を128に設定。
            if 'BeamGovernor' in globals():
                self.governor = BeamGovernor(
                    target_ram_frac=0.75, hard_ram_frac=0.88,
                    target_vram_frac=0.80, hard_vram_frac=0.92,
                    max_beam=128, max_keep=2048,
                    decay_after_ratio=0.80 # 残り20%で減衰開始
                )

        except Exception as e:
            print(f"[Pipeline] WARNING: MemGuard initialization failed: {e}", file=sys.stderr)

    def __del__(self):
        # 終了時のクリーンアップ
        self.stop_monitor()

    def stop_monitor(self):
        # Monitorの安全な停止処理
         if hasattr(self, 'monitor') and self.monitor:
            try:
                if hasattr(self.monitor, 'stop'):
                    self.monitor.stop()
            except Exception: pass
            self.monitor = None

    # Engine Wrappers
    def solve_with_kut30(self, task: ARCTask, budget: float, safe_mode: bool) -> List[CandidateSolution]:
        if safe_mode: return []
        try:
            # KUT30実行時にGovernorとMonitorを渡す
            src, conf, solver_logic = self.make_kut30_candidate(
                task, governor=self.governor, monitor=self.monitor, time_budget=budget
            )
            if conf > 0.0:
                return [CandidateSolution(src, conf, solver_logic)]
        except Exception as e:
             print(f"[Worker {os.getpid()}] ERROR: KUT30 Wrapper FAILED: {e}", file=sys.stderr)
        return []

    def solve_with_kut32(self, task: ARCTask, budget: float, safe_mode: bool) -> List[CandidateSolution]:
        # KUT32は軽量かつメモリ安全チェック内蔵のためsafe_modeでも実行
        try:
            src, conf, solver_logic = self.make_kut32_candidate(task)
            if conf > 0.0: return [CandidateSolution(src, conf, solver_logic)]
        except Exception as e:
             print(f"[Worker {os.getpid()}] ERROR: KUT32 Wrapper FAILED: {e}", file=sys.stderr)
        return []

    # メインのソルバーパイプライン
    def solve_task_internal(self, task: ARCTask, budget: float, safe_mode: bool) -> TaskPredictionsInternal:
        cands = []
        start_time = time.time()

        # --- 時間管理フェーズ (Adaptive Budget Allocation) ---
        # [最適化] 12Hルール対応: 適応的な予算配分と早期脱出戦略 (Fast Engines First)

        # AutoTuneで設定されたソフトキャップを参照 (例: GPU 600s, CPU 180s)
        TASK_SOFT_CAP = CONFIG.AUTOTUNE_CFG.get("TASK_SOFT_CAP_S", 180)
        effective_budget = min(budget, TASK_SOFT_CAP)

        # 新しい予算配分比率 (KUT30重視: 50%)
        BUDGET_RATIOS = {
            'KUT30': 0.50, 'KUT32': 0.10, 'Induction': 0.10,
            'Transduction': 0.10, 'GeomMatch': 0.10, 'ALO': 0.10
        }
        budgets = {k: effective_budget * v for k, v in BUDGET_RATIOS.items()}

        # 早期脱出設定
        EARLY_EXIT_CONFIDENCE = 0.98
        KUT30_REDUCED_RATIO = 0.20 # 早期脱出時、KUT30予算を元の20%に制限
        early_exit_triggered = False

        # Transduction用ヒューリスティクススコア（事前計算）
        try:
            heuristic_scores = self.te.analyze_task_type(task)
        except Exception:
            heuristic_scores = {'color': 0.1, 'fill': 0.1, 'move': 0.1}

        # --- Engine Execution Phase (Fast Engines First) ---
        # [最適化] 高速エンジン群を先に実行し、早期脱出を試みる

        FAST_ENGINES = [
            ('KUT32', self.solve_with_kut32),
            ('ALO', self.alo.solve),
            ('GeomMatch', self.gm.solve),
            ('Induction', self.ie.solve),
            # Transductionは引数が異なるため別途処理
        ]

        for name, solver_fn in FAST_ENGINES:
            # タイムチェック
            if time.time() - start_time > effective_budget: break

            try:
                new_cands = solver_fn(task, budgets[name], safe_mode)
                cands.extend(new_cands)

                # 早期脱出判定 (エンジンの信頼度に基づく)
                if not early_exit_triggered and any(c.confidence >= EARLY_EXIT_CONFIDENCE for c in new_cands):
                    early_exit_triggered = True

            except Exception as e:
                print(f"[Worker {os.getpid()}] ERROR: {name} FAILED: {e}", file=sys.stderr)

        # Engine: Transduction
        if time.time() - start_time <= effective_budget:
            try:
                trans_cands = self.te.solve(task, budgets['Transduction'], heuristic_scores, safe_mode)
                cands.extend(trans_cands)
                if not early_exit_triggered and any(c.confidence >= EARLY_EXIT_CONFIDENCE for c in trans_cands):
                     early_exit_triggered = True
            except Exception as e:
                 print(f"[Worker {os.getpid()}] ERROR: Transduction FAILED: {e}", file=sys.stderr)

        # --- Engine Execution Phase (KUT30) ---

        # KUT30の予算調整
        kut30_budget = budgets['KUT30']
        time_elapsed = time.time() - start_time
        time_left = effective_budget - time_elapsed

        if early_exit_triggered:
            # 早期脱出がトリガーされた場合、KUT30の予算を削減
            reduced_budget = kut30_budget * KUT30_REDUCED_RATIO
            # 残り時間も考慮
            kut30_budget = min(reduced_budget, time_left)
            if kut30_budget > 5: # 5秒以上ある場合のみログ出力
                 print(f"[Pipeline] Early Exit triggered. KUT30 budget reduced to {kut30_budget:.1f}s.", file=sys.stderr)

        # 残り時間が十分にある場合のみKUT30を実行 (最低5秒)
        if time_left >= 5.0 and kut30_budget >= 5.0:
            # solve_with_kut30内でエラーハンドリング済み
            kut30_cands = self.solve_with_kut30(task, kut30_budget, safe_mode)
            cands.extend(kut30_cands)
        elif time_left < 5.0:
             print(f"[Pipeline] INFO: Insufficient time remaining ({time_left:.1f}s). Skipping KUT30.", file=sys.stderr)

        # --- End Engine Execution Phase ---

        return self._generate_predictions(task, cands)

    def _generate_predictions(self, task: ARCTask, cands: List[CandidateSolution]) -> TaskPredictionsInternal:
        # 候補リストから最終的な予測を生成（Arbitratorを使用）
        task_predictions = []
        # ダミー予測（フォールバック用）
        dummy = grid_to_numpy([[0]])

        # test_inputsはARCTaskでNumPy化済み
        for test_input in task.test_inputs:
            if test_input is None or test_input.size == 0:
                task_predictions.append({"attempt_1": dummy, "attempt_2": dummy})
                continue

            # Arbitratorで最終選択
            a1, a2 = self.arb.select_final_submissions(cands, test_input)
            task_predictions.append({"attempt_1": a1, "attempt_2": a2})
        return task_predictions

# ▼▼▼ Wrapper function for gpu_distribute integration ▼▼▼
def solve_task_distributed(task_blob: Dict):
    """
    Wrapper function executed by gpu_distribute worker.
    """
    task_id = task_blob.get('task_id')
    task_data = task_blob.get('task_data')

    # [最適化] デフォルト予算をAutoTune設定(CFG_INJECTED/CONFIG経由)に合わせる
    default_budget = CONFIG.AUTOTUNE_CFG.get("TASK_SOFT_CAP_S", 180.0)
    budget = task_blob.get('budget', default_budget)
    safe_mode = task_blob.get('safe_mode', False)

    if not task_id or not task_data:
        return {'task_id': task_id or 'unknown', 'predictions': [], 'fallback': True}

    # Signal handling (ワーカープロセスでのCtrl+Cなどを無視)
    try:
        if hasattr(signal, 'SIGINT'): signal.signal(signal.SIGINT, signal.SIG_IGN)
    except: pass

    task = None
    solver = None
    try:
        # 1. Initialize Task and Solver
        # ARCTask初期化でデータがNumPy(uint8)化される
        task = ARCTask(task_data, task_id)
        solver = KUT_OS_Solver_Internal()

        # 2. Execute Solver
        predictions_internal = solver.solve_task_internal(task, budget, safe_mode)

        # 3. Normalize predictions (NumPy配列からリストへ変換＋サニタイズ)
        # [最適化] numpy_to_listで形状(30x30)と値(0-9)が保証される。
        final_predictions_list = []
        for pred_dict in predictions_internal:
            final_predictions_list.append({
                "attempt_1": numpy_to_list(pred_dict.get("attempt_1")),
                "attempt_2": numpy_to_list(pred_dict.get("attempt_2"))
            })
        return {'task_id': task_id, 'predictions': final_predictions_list, 'fallback': False}

    except Exception as e:
        # Failsafe mechanism (Input Copy Fallback)
        print(f"CRITICAL ERROR in distributed worker for task {task_id}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        # フォールバック予測（入力をそのままコピーし、安全にサニタイズ）
        fallback_predictions = []
        try:
            test_data = task_data.get("test", [])
            if isinstance(test_data, list) and test_data:
                for item in test_data:
                    # 入力データをnumpy_to_listで安全にサニタイズしてコピー
                    input_grid_sanitized = numpy_to_list(item.get("input", [[0]]))
                    fallback_predictions.append({"attempt_1": input_grid_sanitized, "attempt_2": [[0]]})
            else:
                 fallback_predictions.append({"attempt_1": [[0]], "attempt_2": [[0]]})
        except Exception:
             # フォールバック自体の失敗時
             fallback_predictions = [{"attempt_1": [[0]], "attempt_2": [[0]]}]

        return {'task_id': task_id, 'predictions': fallback_predictions, 'fallback': True}

    finally:
        # [最適化] Cleanup強化: 確実なリソース解放
        if solver:
            # 明示的なMonitor停止メソッド呼び出し
            if hasattr(solver, 'stop_monitor'):
                try: solver.stop_monitor()
                except: pass
            del solver
        if task:
            del task

        # 明示的なガベージコレクション
        gc.collect()

        # GPUメモリの解放（もしCuPy等が将来使われた場合）
        # グローバル変数GPU_ENABLED(ワーカー内で定義)を参照
        if globals().get('GPU_ENABLED', False):
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                # print(f"[Worker {os.getpid()}] INFO: GPU memory freed.", file=sys.stderr)
            except ImportError:
                pass
            except Exception:
                pass
'''

# ==============================================================================
# ARC Prize 2025 ランキング審査 100% 通過版 (OMUX004 Final - バグ修正済)
# 修正点:
# 1. (BugFix) write_submission_to_root: 予測グリッドを出力サイズ基準でリサイズ
# 2. (RuleFix) write_submission_to_root: attempt_1とattempt_2が同一でも両方提出
# ==============================================================================
import hashlib
import importlib.util
import sys
import traceback
import json
import time
import os
import shutil
import threading
import subprocess # GPU検出用
import multiprocessing # CPU数検出・spawn設定用
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

# -------------------------------
# 0. 事前設定: spawnコンテキストの適用 (Pre-setup: Apply spawn context)
# -------------------------------
# [最適化] spawn設定を最優先で実行
if __name__ == "__main__":
    MP_START_METHOD = 'spawn'
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        # メソッドが未設定または異なる場合、force=Trueで設定を試みる
        if current_method != MP_START_METHOD:
            multiprocessing.set_start_method(MP_START_METHOD, force=True)
            print(f"INFO: Multiprocessing start method set to '{MP_START_METHOD}'.")
        else:
             print(f"INFO: Multiprocessing already using '{MP_START_METHOD}'.")
    except RuntimeError as e:
        # すでに開始されている場合など
        print(f"WARNING: Could not set multiprocessing start method: {e}", file=sys.stderr)
    except Exception as e:
         print(f"ERROR: Multiprocessing configuration failed: {e}", file=sys.stderr)

# -------------------------------
# 1. 環境検出と設定 (Environment Detection and Setup)
# -------------------------------
# [最適化] 堅牢な環境検出とAutoTune復元
KAGGLE_ENV = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# パス設定
if KAGGLE_ENV:
    INPUT_BASE_DIR = Path("/kaggle/input")
    # [修正] WORKING_DIRを/tmpから/kaggle/workingに戻す
    WORKING_DIR = Path("/kaggle/working")
    print("Running in Kaggle environment.")
else:
    INPUT_BASE_DIR = Path("./")
    WORKING_DIR = Path("./working")
    print("Running in Local environment.")

WORKING_DIR.mkdir(exist_ok=True, parents=True)

# INPUT_DIR特定ロジック
def detect_input_dir(base_dir: Path) -> Path:
    possible_names = ['arc-prize-2025', 'arc-agi-2024-GCP', 'evaluation_data']
    if base_dir.is_dir():
        for name in possible_names:
            if (base_dir / name).is_dir():
                return base_dir / name
    return base_dir if KAGGLE_ENV else base_dir / 'evaluation_data'
INPUT_DIR = detect_input_dir(INPUT_BASE_DIR)
if not KAGGLE_ENV: INPUT_DIR.mkdir(exist_ok=True, parents=True)

# GPU検出
def has_gpu():
    try:
        if shutil.which("nvidia-smi") is None: return False
        subprocess.check_output(["nvidia-smi","-L"], stderr=subprocess.STDOUT, timeout=5)
        return True
    except: return False
GPU_ENABLED = has_gpu() or (os.environ.get("CUDA_VISIBLE_DEVICES","") != "")

# AutoTune設定 (12H Optimized)
TOTAL_TIME_LIMIT = 12 * 60 * 60
SAFETY_MARGIN = 30 * 60 # 30分
GLOBAL_WALL_S = max(600, TOTAL_TIME_LIMIT - SAFETY_MARGIN)
CFG = {"TASK_SOFT_CAP_S": 180, "MAX_BEAMS": 10} # CPU
if GPU_ENABLED:
    CFG.update({"TASK_SOFT_CAP_S": 600, "MAX_BEAMS": 28}) # GPU

# -------------------------------
# 2. ヘッダー表示
# -------------------------------
print("--- OMUXΩ∞KUT-OS OMUX004 (Fine structure constant α & Collaze fusion Beam - ARC 2025 Robust Env) ---")
print(f"[AutoTune] Limit: {TOTAL_TIME_LIMIT/3600:.2f}h, Wall: {GLOBAL_WALL_S/3600:.2f}h")
print(f"[AutoTune] GPU={GPU_ENABLED} | Beams={CFG['MAX_BEAMS']} | TaskCap={CFG['TASK_SOFT_CAP_S']}s")
print(f"[Environment] Input: {INPUT_DIR}, Working: {WORKING_DIR}")

# パス追加
if str(WORKING_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR.resolve()))

# -------------------------------
# 3. 動的モジュールインポート関数定義
# -------------------------------
# [最適化] ハッシュベースのバージョン管理（変更なし）
def import_dynamic_module(module_base_name: str, code: str, working_dir: Path):
    code_hash = hashlib.sha1(code.encode('utf-8')).hexdigest()[:8]
    module_name = f"{module_base_name}_{code_hash}"
    file_path = working_dir / f"{module_name}.py"

    try:
        should_write = True
        if file_path.exists():
            try:
                if file_path.read_text(encoding='utf-8') == code:
                    should_write = False
            except IOError:
                pass

        if should_write:
            file_path.write_text(code, encoding='utf-8')

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Spec failed: {module_name}")

        if module_name in sys.modules:
            module = sys.modules[module_name]
            if should_write:
                spec.loader.exec_module(module)
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to import dynamic module '{module_name}'.", file=sys.stderr)
        traceback.print_exc()
        raise RuntimeError(f"Dynamic module import failed: {module_name}") from e

# -------------------------------
# 4. GPU_DISTRIBUTE_CODE 定義とロード
# -------------------------------
# [最適化] OMUX004の上書き版コードをベースに、最適化と堅牢化を実施
# CPUワーカー数の最適設定
CPU_COUNT = multiprocessing.cpu_count() or 2
MAX_CPU_WORKERS = 8
N_CPU_WORKERS = max(1, min(MAX_CPU_WORKERS, CPU_COUNT // 2))

GPU_DISTRIBUTE_CODE = f'''
import multiprocessing as mp
import queue
import time
import traceback
import os
import sys
import gc
import shutil
import signal
from typing import Any, Callable, Dict, List, Optional, Tuple

# オプション依存関係
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 定数定義
N_CPU_WORKERS = {N_CPU_WORKERS}

# トップレベル関数定義（Pickle可能）
def _worker_init(gpu_id: int):
    # ワーカー初期化：GPU設定とスレッド数制限
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if HAS_TORCH:
            try:
                torch.cuda.set_device(gpu_id)
            except Exception:
                pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # [最適化] 数値計算ライブラリのスレッド数を制限
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # シグナルハンドリング
    try:
        if hasattr(signal, 'SIGINT'): signal.signal(signal.SIGINT, signal.SIG_IGN)
    except: pass

def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue, gpu_id: int, solve_one: Callable):
    _worker_init(gpu_id)
    while True:
        try:
            # タスク取得
            item = task_queue.get(timeout=5.0)
        except queue.Empty:
            continue
        
        if item is None:
            break
        
        # [重要] itemは(task_id, task_blob)のタプル
        task_id, task_blob = item
        
        try:
            # タスク実行（solve_oneはtask_blobを受け取る）
            # ここで呼び出されるsolve_oneが、worker_wrapperのsafe_solve_task
            res = solve_one(task_blob)
            # 結果返却（エラーなし）
            result_queue.put((task_id, (res, None)))
        except Exception as e:
            # エラーハンドリング
            tb = traceback.format_exc()
            print(f"[Worker GPU:{{gpu_id}}] Error processing {{task_id}}: {{e}}\\n{{tb}}", file=sys.stderr)
            # 結果返却（エラーあり）
            result_queue.put((task_id, (None, str(e))))
        finally:
            # メモリクリーンアップ
            gc.collect()
            if HAS_TORCH and gpu_id >= 0:
                try:
                    torch.cuda.empty_cache()
                except: pass

def distribute(
    tasks: List[Tuple[str, Dict]], # (task_id, task_blob)のリスト
    solve_one: Callable,
    gpu_ids: Optional[List[int]] = None,
    max_inflight_per_gpu: int = 1, # 未使用だがAPI互換性のため維持
    timeout: Optional[float] = None,
) -> Dict[str, Tuple[Optional[Dict], Optional[str]]]:
    
    # [最適化] GPU IDの自動検出ロジック（堅牢化）
    if gpu_ids is None:
        gpu_ids = []
        # 1. Torch
        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    gpu_ids = list(range(torch.cuda.device_count()))
            except: pass
        # 2. nvidia-smi
        if not gpu_ids:
            try:
                if shutil.which("nvidia-smi") is not None:
                    import subprocess
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                        stderr=subprocess.DEVNULL, timeout=5
                    ).decode("utf-8").strip().splitlines()
                    gpu_ids = [int(i.strip()) for i in out if i.strip().isdigit()]
            except: pass
        # 3. CPUモード
        if not gpu_ids:
            gpu_ids = [-1] * N_CPU_WORKERS
            
    print(f"[Distribute] Starting distribution across {{len(gpu_ids)}} Workers (IDs: {{gpu_ids}})")
    
    # [最適化] マルチプロセッシングコンテキスト（メインで設定されたspawnを利用）
    ctx = mp.get_context()
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    # タスク投入
    for task_item in tasks:
        task_queue.put(task_item)
    
    # 終了シグナル投入
    for _ in gpu_ids:
        task_queue.put(None)
    
    # ワーカープロセスの起動
    processes = []
    for gpu_id in gpu_ids:
        # targetにトップレベル関数を指定（Pickle可能）
        p = ctx.Process(target=_worker_loop, args=(task_queue, result_queue, gpu_id, solve_one), daemon=True)
        p.start()
        processes.append(p)
        
    # 結果収集ループ
    results = {{}}
    completed = 0
    N_TASKS = len(tasks)
    start_time = time.time()
    
    while completed < N_TASKS:
        # タイムアウトチェック
        if timeout is not None and time.time() - start_time > timeout:
            print(f"[Distribute] Overall timeout reached. Stopping.", file=sys.stderr)
            break
        
        try:
            # 結果取得
            task_id, (res, err) = result_queue.get(timeout=5.0)
            results[task_id] = (res, err)
            completed += 1
            
            # [最適化] 進捗ログ（ETA表示）
            if completed % len(gpu_ids) == 0 or completed == N_TASKS:
                 elapsed = time.time() - start_time
                 rate = completed / elapsed if elapsed > 0 else 0
                 eta_s = (N_TASKS - completed) / rate if rate > 0 else 0
                 try:
                     eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_s))
                 except:
                     eta_str = f"{{int(eta_s)}}s"
                 print(f"[Distribute] Progress: {{completed}}/{{N_TASKS}} | Elapsed: {{elapsed:.1f}}s | Rate: {{rate:.2f}} t/s | ETA: {{eta_str}}")
                 
        except queue.Empty:
            # ワーカー全滅チェック
            if not any(p.is_alive() for p in processes):
                print(f"[Distribute] CRITICAL: All workers died unexpectedly.", file=sys.stderr)
                break
            continue
            
    # 終了処理
    print(f"[Distribute] Waiting for workers to terminate...")
    for p in processes:
        p.join(timeout=120.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10.0)
        if hasattr(p, 'close'):
            try: p.close()
            except: pass
            
    print(f"[Distribute] Finished. Total time: {{time.time() - start_time:.1f}}s.")
    return results
'''

# モジュールロード
try:
    gpu_distribute = import_dynamic_module('gpu_distribute_v9_robust', GPU_DISTRIBUTE_CODE, WORKING_DIR)
    DISTRIBUTE_FUNCTION = gpu_distribute.distribute
    print(f"INFO: Loaded Robust Distributed Executor ({gpu_distribute.__name__}).")
except Exception as e:
    print("CRITICAL: Failed to load Distributed Executor.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# -------------------------------
# 5. Worker ModuleとWrapperのロード
# -------------------------------
# 
# 【重要】 ここで WORKER_MODULE_CODE を定義する必要があります。
# ステップ2 で提供される「思考エンジン」のコードをここに挿入してください。
#
# (↓ ステップ2で提供するコードをここに貼り付け ↓)
#
# WORKER_MODULE_CODE = """
# (ここにステップ2のコードを挿入)
# """
#
# (↑ ステップ2で提供するコードをここに貼り付け ↑)
#


# Worker Moduleロード (WORKER_MODULE_CODEは定義済み前提)
try:
    # WORKER_MODULE_CODE変数が存在するか確認（前のステップで定義されている必要がある）
    if 'WORKER_MODULE_CODE' not in globals():
         # フォールバック（デバッグ用）
         print("CRITICAL ERROR: WORKER_MODULE_CODE is not defined.", file=sys.stderr)
         print("Please define the WORKER_MODULE_CODE variable (Step 2).", file=sys.stderr)
         # 実行を継続させるため、最小限のダミーを定義
         WORKER_MODULE_CODE = '''
def solve_task_distributed(blob): 
    # ダミーの実装: 入力をそのまま返す
    task_data = blob.get("task_data", {})
    predictions = []
    for test_case in task_data.get("test", []):
        inp = test_case.get("input", [[0]])
        predictions.append({
            "attempt_1": inp,  # 入力をそのまま返す
            "attempt_2": [[0]] # ゼロを返す
        })
    return {"task_id": blob.get("task_id", "unknown"), "predictions": predictions, "fallback": True}
    
def numpy_to_list(g): 
    if hasattr(g, 'tolist'): g = g.tolist()
    if not isinstance(g, list): return [[0]]
    if not g: return [[0]]
    if not isinstance(g[0], list): g = [g]
    return [[int(v) if isinstance(v, (int, float)) else 0 for v in row] for row in g]
'''
         print("WARNING: Using dummy WORKER_MODULE_CODE. This will not solve any tasks.")

    kutos_worker = import_dynamic_module('kutos_worker_v9', WORKER_MODULE_CODE, WORKING_DIR)
    print(f"INFO: Successfully initialized Worker Module ({kutos_worker.__name__}).")
except Exception as e:
    print("CRITICAL: kutos_worker failed to load.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# Wrapper定義とロード
# [最適化] worker_wrapper.py: spawn環境での安全性と堅牢性を確保
# WORKING_DIRを文字列としてコードに埋め込む（子プロセスでのパス解決を確実にする）
WORKING_DIR_STR = str(WORKING_DIR.resolve())
WRAPPER_CODE = f'''
import sys
from pathlib import Path
import importlib.util
import signal
import time
import traceback
from typing import Dict, Any

# [修正] WORKING_DIRをメインプロセスから継承した正しいパスに設定
WORKING_DIR = Path("{WORKING_DIR_STR}")
if str(WORKING_DIR) not in sys.path:
    sys.path.insert(0, str(WORKING_DIR))

# グローバル変数の初期化
SOLVE_TASK_DISTRIBUTED_FUNC = None

# デフォルトのサニタイズ関数（フォールバック用）
def _default_numpy_to_list(g: Any) -> list:
    try:
        if hasattr(g, 'tolist'): g = g.tolist()
        if not isinstance(g, list): return [[0]]
        if not g or not isinstance(g[0], list): 
             g = [g] if isinstance(g, list) and g and isinstance(g[0], (int, float)) else [[0]]
        out = []
        for row in g:
            if not isinstance(row, list): row = [0]
            out.append([max(0, min(9, int(v))) if isinstance(v, (int, float)) else 0 for v in row])
        return out
    except: return [[0]]
WORKER_NUMPY_TO_LIST = _default_numpy_to_list

def reload_worker_module():
    # ワーカーモジュールを動的にロード/リロード (spawn対応)
    global SOLVE_TASK_DISTRIBUTED_FUNC, WORKER_NUMPY_TO_LIST
    try:
        # kutos_worker_v9_*.py を検索
        worker_file = next(WORKING_DIR.glob("kutos_worker_v9_*.py"), None)
        if worker_file:
            module_name = worker_file.stem
            # ファイルパスから直接ロード
            spec = importlib.util.spec_from_file_location(module_name, worker_file)
            if spec and spec.loader:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                # モジュール実行（リロード）
                spec.loader.exec_module(module)
                # 関数ポインタを更新
                SOLVE_TASK_DISTRIBUTED_FUNC = getattr(module, 'solve_task_distributed', None)
                WORKER_NUMPY_TO_LIST = getattr(module, 'numpy_to_list', _default_numpy_to_list)
    except Exception as e:
        print(f"[Wrapper] Error reloading worker module: {{e}}", file=sys.stderr)
    
    # 最終確認とフォールバック
    if SOLVE_TASK_DISTRIBUTED_FUNC is None:
        print(f"[Wrapper] CRITICAL: SOLVE_TASK_DISTRIBUTED_FUNC not found. Using fallback.", file=sys.stderr)
        SOLVE_TASK_DISTRIBUTED_FUNC = lambda blob: {{"task_id": blob.get("task_id", "unknown"), "predictions": [], "fallback": True}}

# タイムアウトハンドラ
def timeout_handler(signum, frame):
    raise TimeoutError("Task exceeded budget")

# [最適化] メインのタスク実行関数（Pickle可能）
def safe_solve_task(task_blob: Dict) -> Dict:
    # [修正] 引数はtask_blob (Dict) を受け取る。タプル展開は不要。
    
    # ワーカーモジュールをリロード（spawn環境で必須だが非効率。改善の余地ありだが安全性優先）
    reload_worker_module()
    
    task_id = task_blob.get("task_id", "unknown")
    try:
        # 予算から安全マージン(5秒)を引く
        task_budget = max(5.0, float(task_blob.get("budget", 600.0)) - 5.0)
    except (ValueError, TypeError):
        task_budget = 600.0

    # タイムアウト設定（UNIX系のみ）
    is_unix = hasattr(signal, 'SIGALRM')
    if is_unix:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max(1, int(task_budget)))
        except Exception:
            is_unix = False
            
    try:
        # タスク実行
        result = SOLVE_TASK_DISTRIBUTED_FUNC(task_blob)
        if isinstance(result, dict):
            return result
        else:
             raise TypeError(f"Worker returned non-dict type: {{type(result)}}")
             
    except TimeoutError:
        print(f"[Wrapper TIMEOUT] Task {{task_id}} exceeded budget {{task_budget:.1f}}s.", file=sys.stderr)
        return {{"task_id": task_id, "predictions": [], "fallback": True, "error": "Timeout"}}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Wrapper ERROR] Task {{task_id}}: {{e}}\\n{{tb}}", file=sys.stderr)
        return {{"task_id": task_id, "predictions": [], "fallback": True, "error": str(e)}}
    finally:
        # タイムアウト解除
        if is_unix:
            try:
                signal.alarm(0)
            except: pass
'''

# Wrapperロード
try:
    wrapper_module = import_dynamic_module('worker_wrapper_v9_robust', WRAPPER_CODE, WORKING_DIR)
    # 分散実行で呼び出す関数ポインタを設定
    safe_solve_task = wrapper_module.safe_solve_task
    print(f"INFO: Successfully initialized Worker Wrapper ({wrapper_module.__name__}).")
except Exception as e:
    print("CRITICAL: worker_wrapper failed to load.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# -------------------------------
# 6. サニタイズ関数 (Sanitization Function)
# -------------------------------
# [最適化] メインスクリプトのサニタイズ関数を定義（Wrapper経由でWorker関数を利用）
MAX_H, MAX_W = 30, 30

def sanitize_and_resize_grid(g: Any, expected_hw: Tuple[int, int]) -> List[List[int]]:
    """
    提出用グリッドデータを厳密にサニタイズし、期待サイズにリサイズする。
    """
    H_expected, W_expected = expected_hw
    H = max(1, min(MAX_H, H_expected))
    W = max(1, min(MAX_W, W_expected))

    # 1. Wrapper経由でワーカーのnumpy_to_listを呼び出し、基本サニタイズ
    try:
        # wrapper_moduleのWORKER_NUMPY_TO_LISTを直接利用
        sanitized_list = wrapper_module.WORKER_NUMPY_TO_LIST(g)
    except Exception as e:
        # 失敗時はゼロ埋めフォールバック
        print(f"WARNING: Sanitization failed during conversion: {e}. Falling back to zero grid.", file=sys.stderr)
        return [[0] * W for _ in range(H)]

    # 2. 厳密なリサイズ（パディング/クロッピング）
    # 高さ調整
    resized_list = sanitized_list[:H]
    if len(resized_list) < H:
        # [修正] パディングする行の幅を W に合わせる
        resized_list += [[0] * W] * (H - len(resized_list))

    # 幅調整と値の最終確認
    for i, row in enumerate(resized_list):
        if not isinstance(row, list): row = []

        if len(row) > W:
            row = row[:W]
        elif len(row) < W:
            row += [0] * (W - len(row))

        # 値の最終確認（保険）。ワーカー側で実施済みだが念のため。
        final_row = []
        for x in row:
            try:
                val = int(x)
                if not (0 <= val <= 9): val = 0
            except:
                val = 0
            final_row.append(val)
        resized_list[i] = final_row

    return resized_list

# -------------------------------
# 7. run_all_tasks (Main Execution Runner)
# -------------------------------
# [最適化] 精緻な時間管理と堅牢な実行
def run_all_tasks(tasks: List[Tuple[str, Dict]], total_runtime_limit: float):
    if not tasks:
        return {}

    if not DISTRIBUTE_FUNCTION:
        sys.exit(1) # 実行不能

    N_TASKS = len(tasks)

    # 時間予算計算
    # SAFETY_MARGIN (デフォルト30分) を利用
    available_time = max(60.0, total_runtime_limit - SAFETY_MARGIN)
    avg_budget = available_time / N_TASKS

    # AutoTuneのソフトキャップを参照
    TASK_SOFT_CAP = float(CFG.get("TASK_SOFT_CAP_S", 180.0))

    # 最終予算 = min(平均予算, ソフトキャップ)。最低10秒確保。
    task_budget = max(10.0, min(avg_budget, TASK_SOFT_CAP))

    print(f"\n>>> Starting distributed execution (OMUX004 Robust Runner)...")
    print(f"    Available time: {available_time/3600:.2f}h. Tasks: {N_TASKS}.")
    print(f"    Allocated budget: {task_budget:.1f}s/task (Avg: {avg_budget:.1f}s, Cap: {TASK_SOFT_CAP:.1f}s).")

    # タスクブロブ準備
    # [重要] (tid, dict) タプル形式。gpu_distributeがこれを処理する。
    blobs = [
        (tid, {
            "task_id": tid,
            "task_data": data,
            "budget": task_budget,
            "safe_mode": False, # [最適化] safe_modeをFalseに（エンジン内部で制御）
        }) for tid, data in tasks
    ]

    # [最適化] プログレスモニタリングはgpu_distribute内部で行うため削除。

    try:
        # 分散実行開始
        # solve_oneにはwrapperのsafe_solve_taskを渡す（Pickle可能）
        results = DISTRIBUTE_FUNCTION(
            tasks=blobs,
            solve_one=safe_solve_task,
            gpu_ids=None, # 自動検出
            timeout=available_time + 180 # 思考時間 + 終了待機マージン
        )
        print(f">>> Execution finished. {len(results)}/{N_TASKS} tasks processed.")

    except Exception as e:
        print(f"CRITICAL ERROR: run_all_tasks failed: {e}", file=sys.stderr)
        traceback.print_exc()
        results = {}

    return results

# -------------------------------
# 8. 提出生成 (Submission Generation)
# -------------------------------
# [最適化] フォーマット厳守と完全性保証 (OMUX003修正版を踏襲)
def write_submission_to_root(results: Dict[str, Tuple[Optional[Dict], Optional[str]]], tasks: List[Tuple[str, Dict]]):
    print(f"\n>>> Generating submission file (ARC Prize 2025 Format - Rigorous Compliance)...")
    submission = {}
    
    # 提出パス（Kaggle標準）
    out_path_root = Path("./submission.json")

    for tid, tdata in tasks:
        # タスクIDはstemを使用（tidがパスの場合も考慮）
        task_id_key = Path(tid).stem
        tests = tdata.get("test", [])
        N_TESTS = len(tests)

        # 結果取得
        res_tuple = results.get(tid)
        if res_tuple:
            res, err = res_tuple
            preds = (res or {}).get("predictions", []) if res else []
            # if err: print(f"    INFO: Task {task_id_key} error: {err[:100]}...")
        else:
            preds = []
            # print(f"    WARNING: No result for task {task_id_key}. Using fallback.")

        task_outs = []
        for i in range(N_TESTS):
            
            # フォールバック用の入力グリッド
            try:
                inp_fallback = tests[i].get("input", [[0]])
                if not isinstance(inp_fallback, list) or not inp_fallback: inp_fallback = [[0]]
            except:
                inp_fallback = [[0]]


            # 予測結果の取得またはフォールバック生成
            if i < len(preds) and isinstance(preds[i], dict):
                pred = preds[i]
                a1_raw = pred.get("attempt_1", inp_fallback) # デフォルトは入力コピー
                a2_raw = pred.get("attempt_2", [[0]]) # デフォルトはゼロ
            else:
                a1_raw = inp_fallback
                a2_raw = [[0]]

            # 
            # 【!! 致命的バグ修正 !!】
            # 期待されるH, W (expected_hw) を、入力(input)ではなく、
            # 予測された attempt_1 (a1_raw) のサイズから決定します。
            #
            grid_to_measure = a1_raw
            if not isinstance(grid_to_measure, list) or not grid_to_measure:
                grid_to_measure = [[0]] # a1が不正な場合のフォールバック

            try:
                if isinstance(grid_to_measure[0], list):
                    H_expected, W_expected = len(grid_to_measure), len(grid_to_measure[0])
                else:
                    H_expected, W_expected = 1, len(grid_to_measure)
                # 最小サイズを1x1に保証
                expected_hw = (max(1, H_expected), max(1, W_expected))
            except:
                expected_hw = (1, 1) # 最終フォールバック
            
            # (旧バグコード: expected_hwは入力(tests[i])から取得されていた)

            # 厳密なサニタイズとリサイズ
            # これで a1 と a2 は、a1 のサイズに正しくリサイズされる
            a1 = sanitize_and_resize_grid(a1_raw, expected_hw)
            a2 = sanitize_and_resize_grid(a2_raw, expected_hw)

            # prediction_id: 0 (必須)
            task_outs.append({"output_id": i, "prediction_id": 0, "output": a1})

            # 
            # 【!! ルール違反修正 !!】
            # attempt_1 と attempt_2 が同一であっても、
            # 「厳密に2試行」のルール(2.4)に従い、prediction_id: 1 を必ず提出します。
            #
            # if a2 != a1: <-- この条件分岐を削除
            task_outs.append({"output_id": i, "prediction_id": 1, "output": a2})

        submission[task_id_key] = task_outs

    # ファイル書き出し
    try:
        json_str = json.dumps(submission, separators=(',', ':'), ensure_ascii=False)

        # カレントディレクトリ(./)に書き込み
        out_path_root.write_text(json_str, encoding='utf-8')

        # WORKING_DIRにもコピー（デバッグ用）
        out_path_working = WORKING_DIR / "submission.json"
        if out_path_working.resolve() != out_path_root.resolve():
             try:
                 shutil.copyfile(out_path_root, out_path_working)
             except: pass

        print(f">>> Successfully wrote submission.json to {out_path_root}.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write submission.json: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            out_path_root.write_text("{}", encoding='utf-8')
        except: pass

# -------------------------------
# 9. タスクロード (Task Loading)
# -------------------------------
# [最適化] 堅牢なタスクローダー（OMUX002最適化版を踏襲）
def load_tasks(input_dir: Path) -> List[Tuple[str, Dict]]:
    print(f"\n>>> Loading evaluation tasks from {input_dir}...")
    tasks = []

    # 標準ファイル名と検索ルート定義
    challenge_files = ['arc-agi_evaluation_challenges.json', 'evaluation_challenges.json', 'test_challenges.json', 'arc-agi_test_challenges.json', 'train.json', 'test.json']
    search_roots = [input_dir]

    # データセット構造対応
    if input_dir.is_dir():
        if (input_dir / 'data' / 'evaluation').is_dir():
            search_roots.append(input_dir / 'data' / 'evaluation')
        # サブディレクトリも探索対象に追加
        try:
            for subdir in input_dir.iterdir():
                if subdir.is_dir() and subdir not in search_roots:
                    search_roots.append(subdir)
        except OSError: pass

    found_path = None

    # Phase 1: 集約ファイルの検索
    for root in search_roots:
        if not root.is_dir(): continue
        for fname in challenge_files:
            p = root / fname
            if p.is_file():
                found_path = p; break
        if found_path: break

    if found_path:
        print(f"    Found aggregated file: {found_path}")
        try:
            data = json.loads(found_path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                for tid in sorted(data.keys()):
                    tdata = data[tid]
                    if isinstance(tdata, dict) and "test" in tdata and "train" in tdata:
                        # [重要] tidは一意性を確保するためパス+キーを使用（Distributeのキーとなる）
                        tasks.append((f"{found_path}#{tid}", tdata))
        except Exception as e:
            print(f"ERROR: Failed to load {found_path}: {e}", file=sys.stderr)

    # Phase 2: 個別ファイルの検索（フォールバック）
    if not tasks and not found_path:
        print("    Searching for individual task files (recursive)...")
        try:
            if input_dir.is_dir():
                files = sorted(input_dir.rglob("*.json"))
                for p in files:
                    try:
                        data = json.loads(p.read_text(encoding='utf-8'))
                        if isinstance(data, dict) and "test" in data and "train" in data:
                            # [重要] tidはファイルパスを使用（Distributeのキーとなる）
                            tasks.append((str(p), data))
                    except: continue
        except Exception: pass

    return tasks

# -------------------------------
# 10. メイン実行ブロック (Main Execution Block)
# -------------------------------
if __name__ == "__main__":
    # spawn設定はスクリプト冒頭で実施済み

    start_time = time.time()

    # 1. タスクロード
    tasks = load_tasks(INPUT_DIR)
    n_tasks = len(tasks)
    print(f">>> Loaded {n_tasks} tasks.")

    # 2. 実行と提出生成
    if n_tasks == 0:
        print("INFO: No tasks found. Generating empty submission.")
        try:
             Path("./submission.json").write_text("{}", encoding='utf-8')
        except: pass
    else:
        # GLOBAL_WALL_S（例: 11.5時間）を制限時間として渡す
        results = run_all_tasks(tasks, GLOBAL_WALL_S)
        # write_submission_to_rootで./submission.jsonへの書き込みを行う
        write_submission_to_root(results, tasks)

    elapsed = time.time() - start_time
    print(f"\n--- OMUXΩ∞KUT-OS OMUX004 (Robust Pipeline) execution finished. ---")
    print(f"Total time: {elapsed/60:.2f} minutes ({elapsed:.1f}s). Limit: {GLOBAL_WALL_S/60:.2f} min.")
    

