hydra:
  run:
    dir: ../stuff/hydra_log/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: ../stuff/hydra_log/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}

model_str: gemma-2-9b-v2
idx: 5
init_sent: 'from and and as and have the in is it of not that the to we with you advent card carol cheer chocolate chimney doll dream drive eat elf family fireplace game give gifts grinch holiday hope holly hohoho jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night nutcracker ornament ornament of the night peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk wish wonder workshop workshop yuletide angel bake beard believe bow candy candle cheer chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting snowglobe stocking wreath wrapping paper'
bs_mul: 10
max_M: 25
gpu_id: 0

temp_start : 0.5
temp_end : 0.025
cooling_rate : 0.4 # 0.7

max_epoch : 30 # 13
patience : 3
steps_per_epoch : 1.2


import os
import random
import time
import socket
import gc
import os
import sys
import math
import time
import csv
from math import exp
from collections import Counter
from itertools import combinations,permutations
from typing import List, Optional, Union
from tqdm import tqdm
import random
import string
from glob import glob
from datetime import datetime
import torch.distributed as dist
import numpy as np
import pandas as pd
import transformers
import torch
import ipynbname
import msgpack
import numpy as np
import pandas as pd
import transformers
import torch
import torch.distributed as dist
import ipynbname
from collections import Counter
import hydra
from pprint import pprint

######################## EXP_ID ########################
# 初始化 EXP_ID 为默认值
EXP_ID = '1201a'

# 优先尝试通过 __file__ 获取脚本名称
try:
    # 只能在 .py 文件中运行
    current_file_path = __file__
    file_name = os.path.basename(current_file_path)
    EXP_ID = file_name.split(".")[0].split("_")[0]
except NameError:
    print("无法找到 py 文件。可能是在非 py 环境中运行。")
    # 如果在 Jupyter Notebook 环境，切换到使用 ipynbname
    try:
        notebook_path = ipynbname.path()  # 获取完整路径
        notebook_name = ipynbname.name()  # 获取文件名（不带路径）
        EXP_ID = notebook_name.split(".")[0].split("_")[0]
        print(f"Notebook 文件路径: {notebook_path}")
        print(f"Notebook 文件名: {notebook_name}")
    except FileNotFoundError:
        print("无法找到 Notebook 文件。可能是在非 Notebook 环境中运行。")
    except Exception as e:
        print(f"发生错误: {e}")

# 最终输出 EXP_ID
print(f"Experiment ID: {EXP_ID}")

if 'rtx4090' in socket.gethostname():
    DIR_PREMODEL = '/data2/all_data/premodel'
elif 'nb' in socket.gethostname():
    DIR_PREMODEL = '/mnt/nfs/dmj/premodel'
else:
    DIR_PREMODEL = '../../premodel'
DEBUG = False


seq2miniseq_dicts = {
    0: {'gingerbread': 0, 'fireplace': 1, 'mistletoe': 2, 'ornament': 3, 'reindeer': 4, 'chimney': 5, 'scrooge': 6,
        'advent': 7, 'family': 8, 'elf': 9},
    1: {'gingerbread': 0, 'fireplace': 1, 'mistletoe': 2, 'ornament': 3, 'reindeer': 4, 'chimney': 5, 'scrooge': 6,
        'advent': 7, 'family': 8, 'drive': 9, 'sleep': 10, 'night': 11, 'laugh': 12, 'walk': 13, 'give': 14, 'jump': 15,
        'bake': 16, 'elf': 17, 'the': 18, 'and': 19},
    2: {'decorations': 0, 'nutcracker': 1, 'yuletide': 2, 'workshop': 3, 'stocking': 4, 'ornament': 5, 'holiday': 6,
        'chimney': 7, 'naughty': 8, 'grinch': 9, 'sleigh': 10, 'jingle': 11, 'gifts': 12, 'cheer': 13, 'carol': 14,
        'polar': 15, 'holly': 16, 'beard': 17, 'magi': 18, 'nice': 19},
    3: {'decorations': 0, 'nutcracker': 1, 'yuletide': 2, 'workshop': 3, 'stocking': 4, 'ornament': 5, 'holiday': 6,
        'chimney': 7, 'naughty': 8, 'grinch': 9, 'sleigh': 10, 'jingle': 11, 'unwrap': 12, 'gifts': 13, 'carol': 14,
        'polar': 15, 'holly': 16, 'beard': 17, 'visit': 18, 'relax': 19, 'magi': 20, 'nice': 21, 'sing': 22,
        'cheer': 24, 'and': 25, 'the': 26, 'eat': 27, 'of': 28, 'is': 29},
    4: {'poinsettia': 0, 'peppermint': 1, 'snowglobe': 2, 'fruitcake': 3, 'chocolate': 4, 'fireplace': 5, 'workshop': 6,
        'greeting': 7, 'wrapping': 8, 'believe': 9, 'hohoho': 10, 'candle': 11, 'eggnog': 12, 'puzzle': 13,
        'wonder': 14, 'season': 15, 'cookie': 16, 'wreath': 17, 'kaggle': 18, 'candy': 19, 'dream': 20, 'peace': 21,
        'merry': 22, 'paper': 23, 'night': 24, 'angel': 25, 'game': 26, 'doll': 27, 'hope': 28, 'card': 29, 'milk': 30,
        'star': 31, 'wish': 32, 'that': 33, 'have': 34, 'with': 35, 'from': 36, 'toy': 37, 'joy': 38, 'bow': 39,
        'the': 40, 'and': 41, 'not': 42, 'you': 43, 'to': 44, 'of': 45, 'in': 46, 'it': 47, 'as': 48, 'we': 49},
    5: {'gingerbread': 0, 'decorations': 1, 'nutcracker': 2, 'poinsettia': 3, 'peppermint': 4, 'mistletoe': 5,
        'snowglobe': 6, 'fruitcake': 7, 'chocolate': 8, 'reindeer': 9, 'yuletide': 10, 'stocking': 11, 'greeting': 12,
        'wrapping': 13, 'scrooge': 14, 'holiday': 15, 'naughty': 16, 'believe': 17, 'fireplace': 19, 'advent': 20,
        'family': 21, 'grinch': 22, 'sleigh': 23, 'jingle': 24, 'unwrap': 25, 'hohoho': 26, 'candle': 27, 'eggnog': 28,
        'puzzle': 29, 'wonder': 30, 'season': 31, 'cookie': 32, 'wreath': 33, 'kaggle': 34, 'ornament': 36,
        'workshop': 38, 'chimney': 40, 'drive': 41, 'sleep': 42, 'laugh': 43, 'gifts': 44, 'carol': 45, 'polar': 46,
        'holly': 47, 'beard': 48, 'visit': 49,
        'relax': 50, 'candy': 51, 'dream': 52, 'peace': 53, 'merry': 54, 'paper': 55, 'angel': 56, 'walk': 57,
        'give': 58, 'jump': 59, 'bake': 60, 'magi': 61, 'nice': 62, 'sing': 63, 'game': 64, 'doll': 65, 'hope': 66,
        'card': 67, 'milk': 68, 'star': 69, 'wish': 70, 'that': 71, 'have': 72, 'with': 73, 'from': 74, 'night': 76,
        'cheer': 78, 'elf': 79, 'eat': 80, 'toy': 81, 'joy': 82, 'bow': 83, 'not': 84, 'you': 85, 'the': 88, 'and': 91,
        'is': 92, 'to': 93, 'in': 94, 'it': 95, 'as': 96, 'we': 97, 'of': 99}
}

miniseq2seq_dicts = {
    0: {0: 'gingerbread', 1: 'fireplace', 2: 'mistletoe', 3: 'ornament', 4: 'reindeer', 5: 'chimney', 6: 'scrooge',
        7: 'advent', 8: 'family', 9: 'elf'},
    1: {0: 'gingerbread', 1: 'fireplace', 2: 'mistletoe', 3: 'ornament', 4: 'reindeer', 5: 'chimney', 6: 'scrooge',
        7: 'advent', 8: 'family', 9: 'drive', 10: 'sleep', 11: 'night', 12: 'laugh', 13: 'walk', 14: 'give', 15: 'jump',
        16: 'bake', 17: 'elf', 18: 'the', 19: 'and'},
    2: {0: 'decorations', 1: 'nutcracker', 2: 'yuletide', 3: 'workshop', 4: 'stocking', 5: 'ornament', 6: 'holiday',
        7: 'chimney', 8: 'naughty', 9: 'grinch', 10: 'sleigh', 11: 'jingle', 12: 'gifts', 13: 'cheer', 14: 'carol',
        15: 'polar', 16: 'holly', 17: 'beard', 18: 'magi', 19: 'nice'},
    3: {0: 'decorations', 1: 'nutcracker', 2: 'yuletide', 3: 'workshop', 4: 'stocking', 5: 'ornament', 6: 'holiday',
        7: 'chimney', 8: 'naughty', 9: 'grinch', 10: 'sleigh', 11: 'jingle', 12: 'unwrap', 13: 'gifts', 14: 'carol',
        15: 'polar', 16: 'holly', 17: 'beard', 18: 'visit', 19: 'relax', 20: 'magi', 21: 'nice', 22: 'sing',
        23: 'cheer', 24: 'cheer', 25: 'and', 26: 'the', 27: 'eat', 28: 'of', 29: 'is'},
    4: {0: 'poinsettia', 1: 'peppermint', 2: 'snowglobe', 3: 'fruitcake', 4: 'chocolate', 5: 'fireplace', 6: 'workshop',
        7: 'greeting', 8: 'wrapping', 9: 'believe', 10: 'hohoho', 11: 'candle', 12: 'eggnog', 13: 'puzzle',
        14: 'wonder', 15: 'season', 16: 'cookie', 17: 'wreath', 18: 'kaggle', 19: 'candy', 20: 'dream', 21: 'peace',
        22: 'merry', 23: 'paper', 24: 'night', 25: 'angel', 26: 'game', 27: 'doll', 28: 'hope', 29: 'card', 30: 'milk',
        31: 'star', 32: 'wish', 33: 'that', 34: 'have', 35: 'with', 36: 'from', 37: 'toy', 38: 'joy', 39: 'bow',
        40: 'the', 41: 'and', 42: 'not', 43: 'you', 44: 'to', 45: 'of', 46: 'in', 47: 'it', 48: 'as', 49: 'we'},
    5: {0: 'gingerbread', 1: 'decorations', 2: 'nutcracker', 3: 'poinsettia', 4: 'peppermint', 5: 'mistletoe',
        6: 'snowglobe', 7: 'fruitcake', 8: 'chocolate', 9: 'reindeer', 10: 'yuletide', 11: 'stocking', 12: 'greeting',
        13: 'wrapping', 14: 'scrooge', 15: 'holiday', 16: 'naughty', 17: 'believe', 18: 'fireplace', 19: 'fireplace',
        20: 'advent', 21: 'family', 22: 'grinch', 23: 'sleigh', 24: 'jingle', 25: 'unwrap', 26: 'hohoho', 27: 'candle',
        28: 'eggnog', 29: 'puzzle', 30: 'wonder', 31: 'season', 32: 'cookie', 33: 'wreath', 34: 'kaggle',
        35: 'ornament', 36: 'ornament', 37: 'workshop', 38: 'workshop', 39: 'chimney', 40: 'chimney', 41: 'drive',
        42: 'sleep', 43: 'laugh', 44: 'gifts', 45: 'carol', 46: 'polar', 47: 'holly', 48: 'beard', 49: 'visit',
        50: 'relax', 51: 'candy', 52: 'dream', 53: 'peace', 54: 'merry', 55: 'paper', 56: 'angel', 57: 'walk',
        58: 'give', 59: 'jump', 60: 'bake', 61: 'magi', 62: 'nice', 63: 'sing', 64: 'game', 65: 'doll', 66: 'hope',
        67: 'card', 68: 'milk', 69: 'star', 70: 'wish', 71: 'that', 72: 'have', 73: 'with', 74: 'from', 75: 'night',
        76: 'night', 77: 'cheer', 78: 'cheer', 79: 'elf', 80: 'eat', 81: 'toy', 82: 'joy', 83: 'bow', 84: 'not',
        85: 'you', 86: 'the', 87: 'the', 88: 'the', 89: 'and', 90: 'and', 91: 'and', 92: 'is', 93: 'to', 94: 'in',
        95: 'it', 96: 'as', 97: 'we', 98: 'of', 99: 'of'}
}


bst_sents = [
 'yuletide the the the of of and and and from to is as in that it we with not you have family season holiday gifts give greeting card wrapping paper bow decorations ornament stocking advent candle wreath holly mistletoe poinsettia angel star magi peace joy cheer merry jingle bake cookie gingerbread candy peppermint chocolate milk eggnog fruitcake nutcracker ornament snowglobe carol sing walk drive visit eat sleep relax unwrap toy doll game puzzle jump laugh cheer wish hope believe wonder dream night night elf reindeer sleigh workshop workshop polar beard hohoho scrooge grinch naughty nice chimney chimney fireplace fireplace kaggle'
,'from the the the of of and and not and to in is you that it with as advent card angel bake beard believe bow candy carol candle cheer cheer chocolate chimney chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace fruitcake game gifts give gingerbread greeting grinch have holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night nutcracker ornament ornament paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk we wish wonder workshop workshop wrapping wreath yuletide'
,'from the the the of of and and and to is as in that it we with not you have beard bow cheer chocolate chimney cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament peppermint polar poinsettia reindeer scrooge sleigh snowglobe star stocking wreath wrapping paper yuletide advent angel bake believe candy carol candle cheer chimney decorations dream drive eat elf family fireplace game night gifts give grinch greeting card holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace puzzle relax season sing sleep toy unwrap visit walk wish wonder workshop workshop'
,'the the the of of and to and and have not you it from as in that is we with advent angel beard bow cheer chocolate chimney doll dream drive eat elf family fireplace game night give gifts grinch holiday hope hohoho jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk wish wonder workshop workshop yuletide bake believe candy carol candle cheer chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'from and and of and the as in the is you that it to not with advent card angel bake beard believe bow candy carol cheer chimney decorations doll dream drive eat elf family fireplace game give gifts greeting grinch have holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night of the night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk we wish wonder workshop workshop yuletide candle cheer chimney chocolate cookie eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'from the the the of of and and not and to in is you that it with as advent angel bake beard believe bow cheer chimney doll dream drive eat elf family fireplace game night give gifts grinch have holiday hope hohoho jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk we wish wonder workshop workshop yuletide candle candy carol cheer chocolate chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'from the the the of of and and not and to in is you that it with as advent angel bake beard believe bow cheer chimney doll dream drive eat elf family fireplace game night give gifts grinch have holiday hope hohoho jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk we wish wonder workshop workshop yuletide candle candy carol cheer chocolate chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'from and and as and have in is it not that the to we with you bow cheer chocolate chimney doll dream drive eat elf family fireplace game give grinch holiday hohoho jingle jump laugh milk night naughty nice night polar reindeer relax sing sleep toy unwrap visit walk wish wonder yuletide advent angel bake beard believe candy candle carol cheer chimney cookie decorations eggnog fireplace fruitcake gifts gingerbread greeting card holly hope joy kaggle merry mistletoe nutcracker ornament ornament of the season peace peppermint poinsettia puzzle scrooge sleigh snowglobe star stocking wreath workshop workshop of the magi wrapping paper'
,'from and and as and have is in it not that the to we with you yuletide cheer advent angel bake beard believe bow candy carol candle cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace chimney fruitcake game give gifts gingerbread grinch greeting card holly hohoho holiday hope jingle jump joy kaggle laugh merry milk mistletoe naughty nice night night of the magi nutcracker ornament ornament of the season peace peppermint polar poinsettia puzzle reindeer relax scrooge sleigh sing sleep snowglobe star stocking toy unwrap visit walk wish wonder workshop workshop wreath wrapping paper'
,'from and and as and have is in it not that the to we with you yuletide cheer advent angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog family fireplace fireplace chimney fruitcake game give gifts gingerbread grinch greeting card holly hohoho holiday hope jingle jump joy kaggle laugh merry milk mistletoe naughty nice night night elf nutcracker ornament ornament of the season peace peppermint polar poinsettia puzzle reindeer relax scrooge sing sleigh sleep snowglobe star stocking toy unwrap visit walk wish workshop workshop of the magi wonder wreath wrapping paper'
,'the the the to is as in it have not you we that of with from advent angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace and chimney fruitcake game give gifts gingerbread grinch greeting card holiday hope holly hohoho jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night of cheer nutcracker ornament ornament and snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep stocking star toy unwrap visit walk wonder workshop and workshop wish wreath wrapping paper yuletide'
,'from and and as and have is in it not that the to we with you yuletide cheer advent angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace chimney fruitcake game give gifts gingerbread grinch greeting card holly hohoho holiday hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night of the night of the nutcracker ornament ornament snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep stocking star toy unwrap visit walk wish wonder workshop workshop wreath wrapping paper'
,'the the the of is of to as in that it we with not you from advent angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace and chimney fruitcake game give gifts gingerbread grinch greeting card have holly hohoho holiday hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night and cheer nutcracker ornament ornament and snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep stocking star toy unwrap visit walk wish wonder workshop workshop wreath wrapping paper yuletide'
,'from and of as in is you that it to not with advent card angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace and the chimney fruitcake game give gifts gingerbread greeting grinch have holly hohoho holiday hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night of cheer nutcracker ornament ornament and the snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep stocking star toy unwrap visit walk we wish wonder workshop the workshop wreath wrapping paper yuletide'
,'from and and as and have the in is it of not that the to we with you advent card carol cheer chocolate chimney doll dream drive eat elf family fireplace game give gifts grinch holiday hope holly hohoho jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night nutcracker ornament ornament of the night peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk wish wonder workshop workshop yuletide angel bake beard believe bow candy candle cheer chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting snowglobe stocking wreath wrapping paper'
,'the the the of of and to and and have not you it from as in that is we with advent angel beard bow cheer chimney doll dream drive eat elf family fireplace game night gifts give grinch holiday hope hohoho jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk wish wonder workshop workshop yuletide bake believe carol candy candle cheer chimney chocolate cookie decorations eggnog fireplace fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'the the the of of and to and and have not you it from as in that is we with advent angel beard bow cheer chocolate chimney doll dream drive eat elf family fireplace game night give gifts grinch holiday hope hohoho jingle jump joy kaggle laugh magi merry milk naughty nice night ornament peace polar puzzle reindeer relax scrooge season sing sleigh sleep star toy unwrap visit walk wish wonder workshop workshop yuletide bake believe candy carol candle cheer chimney cookie decorations eggnog fireplace fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia snowglobe stocking wreath wrapping paper'
,'the the the of is of to as in that it we with not you from advent angel bake beard believe bow candy carol candle cheer cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace and chimney fruitcake game give gifts gingerbread grinch greeting card have holiday hope holly hohoho jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night and sleep nutcracker ornament ornament and snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh stocking star toy unwrap visit walk wish wonder workshop workshop wreath wrapping paper yuletide'
,'from and and the and of as in is it of not that the the to we with you advent card carol chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer scrooge sleigh snowglobe stocking toy unwrap wrapping paper wreath yuletide angel bake beard believe bow candy candle cheer cheer chimney chimney decorations dream drive eat elf family fireplace game gifts give greeting grinch have holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace puzzle relax season sing sleep star visit walk wish wonder workshop workshop'
,'from and and the and of as in is it of not that the the to we with you advent card carol chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath yuletide angel bake beard believe bow candy candle cheer cheer chimney chimney decorations dream drive eat elf family fireplace game gifts give greeting grinch have holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop'
,'from the the the of of and and not and to in is you that it with as advent card angel bake beard believe bow candy carol candle cheer cheer chocolate chimney chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace fruitcake game gifts give gingerbread greeting grinch have holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night night nutcracker ornament ornament paper peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep snowglobe star stocking toy unwrap visit walk we wish wonder workshop workshop wrapping wreath yuletide'
,'from and and as and have is in it not that the to we with you yuletide cheer advent angel bake beard believe bow candy candle carol cheer chocolate chimney cookie decorations doll dream drive eat eggnog elf family fireplace fireplace chimney fruitcake game gifts give gingerbread grinch greeting card holiday holly hohoho hope jingle jump joy kaggle laugh magi merry milk mistletoe naughty nice night of the night of the nutcracker ornament ornament snowglobe peace peppermint polar poinsettia puzzle reindeer relax scrooge season sing sleigh sleep stocking star toy unwrap visit walk wish wonder workshop workshop wreath wrapping paper'
,'from the the the of of and and and to as in is you that it we with have not beard bow cheer chocolate chimney doll dream drive eat elf fireplace game give hope jump laugh milk naughty nice night night polar reindeer relax sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide holiday season advent angel bake believe candy carol candle cheer chimney cookie decorations eggnog family fireplace fruitcake gifts gingerbread grinch greeting card holly hohoho jingle joy kaggle magi merry mistletoe nutcracker ornament ornament peace peppermint poinsettia puzzle scrooge sleigh snowglobe stocking star wreath wrapping paper'
,'from and and the and the of as in is not it of that the to with bake candy card chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia scrooge snowglobe stocking toy unwrap wrapping paper wreath yuletide advent angel beard believe bow candle carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace game gifts give greeting grinch have holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar puzzle reindeer relax season sing sleigh sleep star visit walk we wish wonder workshop workshop you'
,'from and and the and of as in is it not that the to we with you bow candy cheer drive eat family game gifts give have holiday hope jump laugh naughty nice night night peace puzzle relax season sing sleep toy unwrap visit walk wish wonder the of advent angel bake beard believe carol candle cheer chimney chimney chocolate cookie decorations doll dream eggnog elf fireplace fireplace fruitcake gingerbread grinch greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint poinsettia polar reindeer scrooge sleigh snowglobe star stocking wreath workshop workshop wrapping paper yuletide'
,'from and and the and of as in is it not of that the to we with you believe bow cheer chocolate drive eat family game gifts give have holiday hope jump laugh naughty nice night night peace puzzle relax season sing sleep toy unwrap visit walk wish wonder the advent angel bake beard candy carol candle cheer chimney chimney cookie decorations doll dream eggnog elf fireplace fireplace fruitcake gingerbread grinch greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer scrooge sleigh snowglobe star stocking wreath workshop workshop wrapping paper yuletide'
,'from the the of the of and and not and to in is you that it with as bake candy card chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath yuletide advent angel beard believe bow candle carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace game gifts give greeting grinch have holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk we wish wonder workshop workshop'
,'from the the the of of and and and to is as in that it we with not you have bake believe cheer drive eat family game give grinch holiday hope jump laugh naughty nice night night peace puzzle relax scrooge season sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide advent angel beard bow candy candle carol cheer chimney chimney chocolate cookie decorations doll dream eggnog elf fireplace fireplace fruitcake gifts gingerbread greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer sleigh snowglobe star stocking wreath wrapping paper'
,'from the the the of of and and and to is as in that it we with not you have bake cheer drive dream eat family game give grinch holiday hope jump laugh naughty nice night night peace puzzle relax scrooge season sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide advent angel beard believe bow candy candle carol cheer chimney chimney chocolate cookie decorations doll eggnog elf fireplace fireplace fruitcake gifts gingerbread greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer sleigh snowglobe star stocking wreath wrapping paper'
,'from the the the of of and and and to in is you we that it have not with as bake believe dream drive eat family game give grinch holiday hope jump laugh naughty nice night night peace puzzle relax scrooge season sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide advent angel beard bow candy carol candle cheer cheer chimney chimney chocolate cookie decorations doll eggnog elf fireplace fireplace fruitcake gifts gingerbread greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer sleigh snowglobe star stocking wreath wrapping paper'
,'from the the the of of and and and to is as in that it we with not you have bake believe cheer dream drive eat family game give grinch holiday hope jump laugh naughty nice night night peace puzzle relax scrooge season sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide advent angel beard bow candy candle carol cheer chimney chimney chocolate cookie decorations doll eggnog elf fireplace fireplace fruitcake gifts gingerbread greeting card holly hohoho jingle joy kaggle magi merry milk mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer sleigh snowglobe star stocking wreath wrapping paper'
,'from the the the of of and and and to is as in that it we with not you have bake candy card chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath yuletide advent angel beard believe bow candle carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace game gifts give greeting grinch holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop'
,'from the the the of of and and and to is as in that it we with not you have bake candy candle card chocolate cookie doll eggnog fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath yuletide advent angel beard believe bow carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace fireplace game give gifts grinch greeting holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop'
,'from the the the of of and and and to as in is you that it we have not with bake believe cheer drive dream eat family game give grinch holiday hope jump laugh milk naughty nice night night peace puzzle relax scrooge season sing sleep toy unwrap visit walk wish wonder workshop workshop yuletide advent angel beard bow candy candle carol cheer chimney chimney chocolate cookie decorations doll eggnog elf fireplace fireplace fruitcake gingerbread gifts greeting card holly hohoho jingle joy kaggle magi merry mistletoe nutcracker ornament ornament peppermint polar poinsettia reindeer sleigh snowglobe star stocking wreath wrapping paper'
,'from the the the of of to is as in that it we with not you have bake candy candle chocolate cookie doll eggnog fruitcake gingerbread greeting card holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath and and and advent angel beard believe bow carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace fireplace game give gifts grinch holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop yuletide' # 29.2650
,'from the the the of of to is as in that it we with not you have beard bow candy chocolate cookie doll eggnog fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath and and and advent angel bake believe carol candle cheer cheer chimney chimney decorations dream drive eat elf family fireplace fireplace game give gifts grinch holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night ornament peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop yuletide' # 29.1746
,'from the the the of of to is as in that it we with not you have beard bow cheer chocolate cookie doll eggnog fruitcake gingerbread greeting card holly mistletoe nutcracker ornament peppermint candy poinsettia scrooge snowglobe stocking unwrap wrapping paper wreath and and and advent angel bake believe carol candle cheer chimney chimney decorations dream drive eat elf family fireplace fireplace game gifts give grinch holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night ornament peace polar puzzle reindeer relax season sing sleigh sleep star toy visit walk wish wonder workshop workshop yuletide' # 29.1667
,'from the the the of of to is as in that it we with not you have bake candy card game chocolate cookie doll eggnog fireplace fruitcake gingerbread holly mistletoe nutcracker ornament ornament peppermint poinsettia puzzle scrooge snowglobe stocking toy unwrap wrapping paper wreath and and and advent angel beard believe bow candle carol cheer cheer chimney chimney decorations dream drive eat elf family fireplace gifts give greeting grinch holiday hohoho hope jingle jump joy kaggle laugh magi merry milk naughty nice night night peace polar reindeer relax season sing sleigh sleep star visit walk wish wonder workshop workshop yuletide' # 28.5322
]


def extract_edges(sequence):
    """
    提取序列中的所有相邻边
    :param sequence: list 序列
    :return: list of tuples，每个元组表示一条边
    """
    return [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]



class NeighborGenerator:
    def __init__(self, N, max_M=None):
        # 10  10  235
        # 20  20  1945
        # 30  30  6630     2000

        # 50 1 3577
        # 50 2 6867
        # 50 3 9882
        # 50 4 12634
        # 50 5 15135
        # 50 10 24295
        # 50 20 30690

        # 100 1 14652
        # 100 2 28717
        # 100 3 42207
        if max_M is None:
            if N == 10:
                max_M = 10
            elif N == 20:
                max_M = 20
            elif N < 30:
                max_M = N
            elif N > 30 and N < 80:
                max_M = 20
            elif N > 80:
                max_M = 3

        origin_seq = list(range(N))

        insertm_seqs = []
        for M in range(1, max_M + 1):
            for i in range(N - M + 1):
                subsequence = origin_seq[i:i + M]
                for j in range(N + 1):
                    if j < i:
                        tweak_seq = origin_seq[:j] + subsequence + origin_seq[j:i] + origin_seq[i + M:]
                        insertm_seqs.append(tweak_seq)
                    elif j > i + M:
                        tweak_seq = origin_seq[:i] + origin_seq[i + M:j] + subsequence + origin_seq[j:]
                        insertm_seqs.append(tweak_seq)
                    else:
                        continue

        swapm_seqs = []
        for M1 in range(1, max_M + 1):
            for M2 in range(1, max_M + 1):
                if (M1>3) & (M2>3):
                    continue
                for i in range(N - M1 + M2 + 1):
                    for j in range(i + M1, N - M2 + 1):
                        tweak_seq = (origin_seq[:i] + origin_seq[j:j + M2] +
                                     origin_seq[i + M1:j] + origin_seq[i:i + M1] + origin_seq[j + M2:])
                        swapm_seqs.append(tweak_seq)

        shufflem_seqs = []
        M = 5
        for i in range(N - M + 1):
            subsequence = origin_seq[i:i + M]
            for s in list(permutations(subsequence)):
                origin_seq_ = origin_seq[:]
                origin_seq_[i:i + M] = s
                shufflem_seqs.append(origin_seq_)

        self.tweak_ops = np.unique(np.array(insertm_seqs+swapm_seqs+shufflem_seqs), axis=0).astype(np.int8)
        print('Number of tweak operations:', len(self.tweak_ops))

    def generate_neighbors(self, current, k):
        shuffled_tweak_ops = self.tweak_ops.copy()
        np.random.shuffle(shuffled_tweak_ops)
        ops = shuffled_tweak_ops[:k]
        neighbors = np.array(current)[ops]
        neighbors = np.unique(neighbors, axis=0).tolist()
        return neighbors

    def gen_all_neib(self, current):
        neighbors = np.array(current)[self.tweak_ops]
        # neighbors = np.unique(neighbors, axis=0).tolist()
        return neighbors



def human_time(start_time):
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), seconds


def convert_to_dense(integer_list, N=100, n=7):
    """
    将一个包含N个0-N之间整数的列表转换为新的整数列表。
    步骤：
    1. 将每个数字转换为n位二进制。
    2. 拼接所有n位二进制数。
    3. 按8位分割，必要时在末尾补零。
    4. 将每个8位分割转换为对应的整数。

    参数:
        integer_list (list of int): 原始整数列表，每个元素在0到31之间。
    返回:
        dense_bytes: 转换后的bytes串。
    """

    # 步骤2：将列表转换为单一整数（通过位操作拼接5位二进制）
    concatenated_int = 0
    for num in integer_list:
        concatenated_int = (concatenated_int << n) | num

    # 计算总位数和需要补充的零位数
    bit_length = N * n  # 30个数字，每个5位，总计150位
    padding = (8 - (bit_length % 8)) % 8  # 计算需要补充的零位数
    concatenated_int <<= padding  # 在末尾补充零

    # 计算总字节数
    total_bits_padded = bit_length + padding
    total_bytes = total_bits_padded // 8  # 150 + 2 = 152位，19个字节

    # 步骤3：将拼接后的整数转换为字节，使用大端字节序
    dense_bytes = concatenated_int.to_bytes(total_bytes, byteorder='big')

    # # 步骤4：将字节转换为整数列表
    # dense_list = list(bytes_padded)
    return dense_bytes


def convert_to_original_list(byte_str, N=100, n=7):
    bytes_padded = byte_str

    # 将字节串转换为单一整数，使用大端字节序
    concatenated_int = int.from_bytes(bytes_padded, byteorder='big')

    # 计算总位数和填充的零位数
    bit_length_padded = len(list(byte_str)) * 8
    bit_length_original = N * n
    padding = bit_length_padded - bit_length_original

    # 去除填充的零位
    concatenated_int >>= padding

    # 按5位分割，恢复原始数字列表
    original_list = []
    for _ in range(N):
        # 取出低5位并将其添加到列表
        num = concatenated_int & ((1 << n) - 1)  # 使用 (1 << n) - 1 获取5个有效位
        original_list.append(num)
        # 右移n位以继续获取下一个数字
        concatenated_int >>= n

    # 恢复的顺序是从低位到高位，需要反转
    original_list.reverse()
    return original_list


# def perturb(seq, rank, side):
#     seq = seq[:]
#     shuffle_length = round(2 * 1.7 ** rank)
#
#     if side == 0:
#         beg_idx = len(seq) - shuffle_length
#         end_idx = len(seq)
#     elif side == 1:
#         shuffle_length = shuffle_length*2
#         beg_idx = int((len(seq) - 1) / 2) - int((shuffle_length - 0.5) / 2)
#         end_idx = int((len(seq) - 1) / 2) + int(round((shuffle_length - 0.5) / 2)) +1
#     elif side == 2:
#         shuffle_length = shuffle_length*2
#         beg_idx = 0
#         end_idx = shuffle_length
#
#     if shuffle_length >= len(seq):
#         beg_idx = 0
#         end_idx = len(seq)
#
#     sub_seq = seq[beg_idx:end_idx]
#     seq[beg_idx:end_idx] = random.sample(sub_seq, len(sub_seq))
#     print(f"Per_side: {side}, Per_rank: {rank},  subseq length: {shuffle_length}, beg_idx: {beg_idx}, end_idx: {end_idx}")
#     return seq, shuffle_length

def perturb(seq, rank):
    seq = np.array(seq)
    sel = random.sample(range(len(seq)), rank)

    sub_seq = seq[sel].copy()
    seq[sel] = random.sample(list(sub_seq), len(sub_seq))

    print(f"Per_rank: {rank},  subseq length: {len(sub_seq)}, sel: {sel}")
    return list(seq)

def read(path):
    start_time = time.time()
    print("Begin reading...")
    obj = msgpack.unpack(open(path, "rb"), strict_map_key=False)
    hours, minutes, seconds = human_time(start_time)
    print(f"Reading time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    return obj


def atom_write(obj, path):
    start_time = time.time()
    print("Begin writing...")
    random_letters = ''.join(random.choices(string.ascii_letters, k=6))
    path_writing = path + f'.writing.{random_letters}'
    msgpack.pack(obj, open(path_writing, "wb"), use_bin_type=True)
    os.rename(path_writing, path)
    hours, minutes, seconds = human_time(start_time)
    print(f"Writing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


class ILS:
    def __init__(self, model_path, device, idx, N, max_M, use_compile=False):
        self.idx = idx
        self.device = device
        self.seq2miniseq_dicts = seq2miniseq_dicts
        self.miniseq2seq_dicts = miniseq2seq_dicts

        # Initialize model and tokenizer
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{self.device}",
        )
        # idx_N_dict = {0:10, 1:20, 2:20, 3:30, 4:50, 5:100}
        idx_bs_dict = {0:70, 1:45, 2:35, 3:27, 4:16, 5:8} # no compile 4090
        if use_compile:
            idx_bs_dict = {0:70, 1:30, 2:30, 3:18, 4:9, 5:5}  # compile 4090
            self.model = torch.compile(self.model, mode="max-autotune", backend="inductor")
            #self.model = torch.compile(self.model, backend="tensorrt")
        self.model = self.model.eval()

        # Initialize other attributes Will be initialized later
        self.N = N
        self.n = int(np.ceil(np.log2(self.N)))
        self.ng = NeighborGenerator(self.N, max_M)
        self.bs = idx_bs_dict[idx]  # round(700/N)  # bs for RTX4090
        self.score_cache = {}
        self.ban_list = []

        self.distance_matrix = np.ones((102,102)) * 35
        self.count_matrix = np.ones((102,102))
        bst_miniseqs = [[100]+self.seq2miniseq(sent.split(' '))+[101] for sent in bst_sents]
        bst_miniedges = [extract_edges(miniseq) for miniseq in bst_miniseqs]
        edges = sum(bst_miniedges,[])
        n0,n1 = zip(*edges)
        self.distance_matrix[n0,n1] = 30.5
        self.paths = np.ones((self.ng.tweak_ops.shape[0],102), dtype=np.int8)
        print('avg_dist<=31:',(self.distance_matrix<=31).sum(),'!!!!')

    def seq2miniseq(self, seq):
        return [self.seq2miniseq_dicts[self.idx][e] for e in seq]

    def miniseq2seq(self, miniseq):
        return [self.miniseq2seq_dicts[self.idx][e] for e in miniseq]

    def cal_ppl(self, sentences, bs=4, verbose=True):
        all_ppl = []
        with torch.no_grad():
            with tqdm(total=len(sentences), desc="Calculating Perplexity", ncols=80, disable=not verbose) as pbar:
                for n, i in enumerate(range(0, len(sentences), bs)):
                    batch = sentences[i:i + bs]
                    batch = [f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in batch]

                    model_inputs = self.tokenizer(
                        batch,
                        return_tensors='pt',
                        add_special_tokens=False,
                        padding=True,
                    )

                    model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

                    outputs = self.model(**model_inputs, use_cache=False)
                    logits = outputs['logits']

                    labels = model_inputs['input_ids']
                    labels[labels == self.tokenizer.pad_token_id] = self.pad_token_label_id
                    labels[labels == self.tokenizer.bos_token_id] = self.pad_token_label_id

                    valid_lengths = (labels != self.pad_token_label_id).sum(dim=-1)

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    loss = self.loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    loss = loss.view(len(batch), -1)
                    sentence_loss = torch.sum(loss, dim=-1) / valid_lengths

                    perplexities = [exp(l.item()) for l in sentence_loss]
                    all_ppl.append(perplexities)

                    if n % 20 == 0:
                        pbar.set_postfix({"min": np.concatenate(all_ppl).min()})
                    pbar.update(len(batch))

        all_ppl = np.concatenate(all_ppl).tolist()
        return all_ppl

    def cal_ppl_miniseqs(self, miniseqs, bs=4, verbose=False):
        sents = [' '.join(self.miniseq2seq(miniseq)) for miniseq in miniseqs]
        return self.cal_ppl(sents, bs, verbose)

    def multi_gpu_infer(self, chunks=None, gather_list=None, verbose=False):
        local_rank = int(os.environ['LOCAL_RANK'])
        local_data = [None]
        dist.scatter_object_list(local_data, chunks, src=0)
        local_data = local_data[0]
        res = self.cal_ppl_miniseqs(local_data, bs=self.bs, verbose=verbose)
        dist.gather_object(res, gather_list, dst=0)
        dist.barrier()

    def rank0_multi_gpu_infer(self, data=None, verbose=False):
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()
        task_flag = torch.tensor(1, device=self.device)
        dist.broadcast(task_flag, src=0)
        len_data = len(data)
        data = data + [data[0].copy() for _ in range(world_size)]
        chunk_size = int(np.ceil(len_data / world_size))
        chunks = [data[i*chunk_size:i*chunk_size + chunk_size] for i in range(world_size)]
        gather_list = [None for _ in range(world_size)]
        self.multi_gpu_infer(chunks=chunks, gather_list=gather_list, verbose=verbose)
        final_results = [item for sublist in gather_list for item in sublist]
        return final_results[:len_data]

    def setup_ddp(self):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.device)

    def cleanup_ddp(self):
        dist.destroy_process_group()

    def read_update_write(self, score_cache, dir_save):
        formatted_date = datetime.now().strftime("%Y%m%d")[-6:]
        glob_str = f'{dir_save}/score_cache_{self.idx}*.msgpack'
        cur_path = f'{dir_save}/score_cache_{self.idx}_{formatted_date}.msgpack'
        paths = glob(glob_str)
        paths.sort()
        if len(paths) == 0:
            print('Writing to', cur_path)
            atom_write(score_cache, cur_path)
        else:
            print('Score cache already exists', paths[-3:])
            old_score_cache = read(paths[-1])
            len_old_score_cache = len(old_score_cache)
            print('Old score cache:', len_old_score_cache)
            old_score_cache.update(score_cache)
            score_cache = old_score_cache
            if len(score_cache) > len_old_score_cache:
                print('Updated score cache:', len(old_score_cache))
                print('Writing to', cur_path)
                atom_write(old_score_cache, cur_path)
        return score_cache

    def sa_search(self, miniseq, temp_start, temp_end, cooling_rate, steps_per_temp, ng,
                  bs_mul=1, full_k=False, verbose=False):
        best_miniseq = current = miniseq.copy()
        delta = 0.02
        best_score = current_score = self.cal_ppl_miniseqs([current], bs=self.bs)[0] + delta

        score_cache = self.score_cache
        temp = temp_start

        k = len(ng.tweak_ops) if full_k else min(int(self.bs * bs_mul), len(ng.tweak_ops))  # *20 * 64*10
        print(f"idx={self.idx}, N={self.N}, k={k}, bs={self.bs}")
        print(f"Start Temperature: {temp:.2f}")
        print(f"Initial score(+{delta:.2f}): {current_score:.4f}, origin: {current_score - delta:.4f}")
        print(current)
        self.num_neighbors = 0
        self.num_cached = 0

        while temp > temp_end:
            for _ in range(steps_per_temp):
                cached_keys = []
                cached_neighbors = []
                cached_scores = []

                uncached_keys = []
                uncached_neighbors = []
                uncached_scores = None

                self.paths[:,1:-1] = ng.gen_all_neib(current)
                distances = np.sum(self.distance_matrix[self.paths[:, :-1], self.paths[:, 1:]], axis=1)
                sorted_idx = np.argsort(distances)

                seled_idx = np.concatenate([sorted_idx[:int(5*k/6)],np.random.choice(sorted_idx[int(5*k/6):], k//6, replace=False) ])
                neighbors = np.unique(self.paths[:,1:-1][seled_idx],axis=0).tolist()

                neighbor_keys = []
                neighbors_ = []
                for neighbor in neighbors:
                    if neighbor not in self.ban_list:
                        neighbor_keys.append(convert_to_dense(neighbor, self.N, self.n))
                        neighbors_.append(neighbor)
                neighbors = neighbors_
                self.num_neighbors += len(neighbor_keys)

                for idx, key in enumerate(neighbor_keys):
                    if key in score_cache:
                        cached_keys.append(key)
                        cached_neighbors.append(neighbors[idx])
                        cached_scores.append(score_cache[key])
                    else:
                        uncached_keys.append(key)
                        uncached_neighbors.append(neighbors[idx])
                self.num_cached += len(cached_keys)

                if uncached_keys:
                    uncached_scores = self.rank0_multi_gpu_infer(uncached_neighbors, verbose)
                    for key, score in zip(uncached_keys, uncached_scores):
                        score_cache[key] = score
                    cached_neighbors.extend(uncached_neighbors)
                    cached_scores.extend(uncached_scores)

                    if min(uncached_scores) < 31:
                        old = (self.distance_matrix<=31).sum()

                        min_neighbors = [l1 for l1, l2 in zip(uncached_neighbors, uncached_scores) if l2 < 31]
                        min_scores = [l1 for l1 in uncached_scores if l1 < 31]
                        bst_miniseqs = [[100]+nei+[101] for nei in min_neighbors]
                        bst_miniedges = [extract_edges(miniseq) for miniseq in bst_miniseqs]

                        count_arr = np.ones(101*len(bst_miniedges))
                        score_arr = np.repeat(np.array(min_scores),101)
                        edges_arr = np.array(sum(bst_miniedges,[]))
                        unique_edges, inverse_indices = np.unique(edges_arr, return_inverse=True, axis=0)
                        grouped_count = np.bincount(inverse_indices, weights=count_arr)
                        grouped_score = np.bincount(inverse_indices, weights=score_arr)

                        n0,n1 = zip(*unique_edges)
                        self.distance_matrix[n0,n1] = (self.distance_matrix[n0,n1] * self.count_matrix[n0,n1] + grouped_score) / (self.count_matrix[n0,n1] + grouped_count)
                        self.count_matrix[n0,n1] = self.count_matrix[n0,n1] + grouped_count

                        if (self.distance_matrix<=31).sum() > old:
                            print('avg_dist<=31:',(self.distance_matrix<=31).sum(),'!!!!')

                    # if max(uncached_scores) > 70:
                    #
                    #     min_neighbors = [l1 for l1, l2 in zip(uncached_neighbors, uncached_scores) if l2 > 70]
                    #     bst_miniseqs = [[100]+nei+[101] for nei in min_neighbors]
                    #     bst_miniedges = [extract_edges(miniseq) for miniseq in bst_miniseqs]
                    #
                    #     edges_arr = np.array(sum(bst_miniedges,[]))
                    #     unique_edges, inverse_indices = np.unique(edges_arr, return_inverse=True, axis=0)
                    #
                    #     n0, n1 = zip(*unique_edges)
                    #     self.distance_matrix[n0,n1] = self.distance_matrix[n0,n1] * 1.005

                sorted_neighbors = [x for _, x in sorted(zip(cached_scores, cached_neighbors))]
                sorted_scores = sorted(cached_scores)  ##
                    # currents = sorted_neighbors[:10]

                first_neighbor_score = sorted_scores[0]
                first_neighbor = sorted_neighbors[0]

                sel_idx = random.choice(range(0, min(20, len(sorted_neighbors))))
                second_neighbor = sorted_neighbors[sel_idx]
                second_neighbor_score = sorted_scores[sel_idx]

                delta_score = first_neighbor_score - current_score
                cond1 = delta_score < 0
                cond2 = random.random() < np.exp(-min(delta_score / temp, 100))

                if cond1 or cond2:
                    if first_neighbor_score < best_score:
                        current = first_neighbor
                        current_score = first_neighbor_score
                        best_miniseq = current.copy()
                        best_score = current_score
                        print(f"》{current_score:.2f}", end="", flush=True)
                    elif cond1:
                        current = first_neighbor
                        current_score = first_neighbor_score
                        print(f">{current_score:.2f}", end="", flush=True)
                    else:
                        current = second_neighbor
                        current_score = second_neighbor_score
                        print(f"<{current_score:.2f}", end="", flush=True)
                    self.ban_list.append(current[:])
                else:
                    print("-", end="", flush=True)

            temp *= cooling_rate
            if verbose:
                print(f"\nTemperature: {temp:.2f}, Current score: {current_score:.2f}")

        print(f"\nFinal score: {best_score:.4f}")
        print(f"num_neighbors: {self.num_neighbors}, num_cached: {self.num_cached}, ratio: {self.num_cached / self.num_neighbors:.2f}")
        return best_miniseq, best_score

    def repeat_sa_search(self, miniseq, globe_bst, n_search, temp_start, temp_end, cooling_rate,
                         max_epoch, patience, steps_per_temp_factor, bs_mul=1, verbose=False):
        start_time = time.time()
        local_bst_miniseq = miniseq[:]
        local_bst_score = self.cal_ppl_miniseqs([miniseq], bs=self.bs)[0]
        globe_bst_score, globe_bst_miniseq = globe_bst
        n_no_better = 0

        for i in range(max_epoch):
            start_time2 = time.time()
            start_len_score_cache = len(self.score_cache)
            cur_patience = max(min(round((14 - i) / patience), 3), 1)
            steps_per_temp = round(2 * steps_per_temp_factor ** i)
            print("+" * 30, f'#{n_search} +{i + 1}/{max_epoch} steps_per_temp: {steps_per_temp}', "+" * 30)
            cur_miniseq, cur_score = self.sa_search(local_bst_miniseq, temp_start, temp_end,
                                                    cooling_rate, steps_per_temp,
                                                    ng=self.ng, bs_mul=bs_mul, verbose=verbose)
            cur_score = self.cal_ppl_miniseqs([cur_miniseq], bs=self.bs)[0]

            if cur_score < local_bst_score:
                local_bst_score = cur_score
                local_bst_miniseq = cur_miniseq[:]
                n_no_better = 0
                print(f"Local update: {cur_score:.4f}, {' '.join(self.miniseq2seq(local_bst_miniseq))}")
                if cur_score < globe_bst_score:
                    print("Global update!!!!!")
                    globe_bst_score = cur_score
                    globe_bst_miniseq = cur_miniseq
                print()
            else:
                n_no_better += 1

            print(f"Global best : {globe_bst_score:.4f}, {' '.join(self.miniseq2seq(globe_bst_miniseq))}")
            hours, minutes, seconds = human_time(start_time)
            print(f"Running time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            print(f"{len(self.score_cache)} keys have been cached")
            print(f"Current score_cache growth rate: {(len(self.score_cache) - start_len_score_cache) / (time.time() - start_time2):.2f}")

            print(f'Patience: {n_no_better}/{cur_patience}')
            if n_no_better >= cur_patience:
                print('Run out of patience, early stop...')
                break

        return local_bst_miniseq, local_bst_score


def save_top_100_keys_to_csv(data_dict, csv_file_path, ils):
    """
    根据字典的 value 倒序排序，取出前100个 key，将 key, value 写入 csv 文件。
    :param data_dict: Python 字典，例如 {'a':10, 'b':5, 'c':20}。
    :param csv_file_path: 输出的 CSV 文件路径。
    """
    # 1. 根据 value 对字典进行倒序排序
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1])

    # 2. 取出前 100 个 key-value 对
    top_100_items = sorted_items[:100]
    top_100_items = [(item[1], ' '.join(ils.miniseq2seq(convert_to_original_list(list(item[0]),ils.N,ils.n)))) for item in top_100_items]
    # 3. 写入 CSV 文件
    #    假设每行保存为: key,value
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头(可选)
        writer.writerow(['value', 'key'])

        # 写入前100项
        for value,key in top_100_items:
            writer.writerow([value, key])
    print(pd.DataFrame(top_100_items))


@hydra.main(config_path="./config", config_name="txt5.yaml", version_base=None)
def run_one_exp(cfg=None):
    cfg = dict(cfg)
    init_sent = cfg['init_sent']
    idx = cfg['idx']
    bs_mul = cfg['bs_mul'] * 64
    max_M = cfg['max_M']
    del cfg['init_sent']
    pprint(cfg)

    ######################## ENV ########################
    DIR_MODEL = f"{DIR_PREMODEL}/{cfg['model_str']}"   # 需要修改
    DIR_SSUM = "./sample_submission.csv"
    DIR_OUTPUT = f'../stuff/output/{EXP_ID}'
    os.makedirs(DIR_OUTPUT, exist_ok=True)

    # 设置环境变量
    #os.environ["CUDA_VISIBLE_DEVICES"] = f'0,1,2,3'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(15)

    # 初始化分布式环境
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    torch.set_float32_matmul_precision('high')

    # 设置 DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = False
    # 读取提交文件
    solution = pd.read_csv(DIR_SSUM)
    submission = pd.read_csv(DIR_SSUM)
    if len(Counter(solution.iloc[idx,1].split(' ')) - Counter(init_sent.split(' '))) != 0:
        print(f"warning len of sent is {len(init_sent.split(' '))}")

    ######################## MODEL ########################
    # 创建 ILS 对象
    ils = ILS(
        model_path=DIR_MODEL,
        device=local_rank,
        idx=idx,
        N=len(init_sent.split(' ')),
        max_M=max_M
    )

    ######################## CFG ########################
    temp_start = cfg['temp_start']                # 初始温度
    temp_end = cfg['temp_end']                    # 结束温度
    cooling_rate = cfg['cooling_rate']            # 冷却率

    max_epoch = cfg['max_epoch']                  # 最大迭代次数
    patience = cfg['patience']                    # 耐心值
    steps_per_epoch = cfg['steps_per_epoch']
    ramdom_tag = ''.join(random.choices(string.ascii_letters, k=6))
    pre_score = 10000

    if rank == 0:
        print('world_size:', dist.get_world_size())
        perturb_rank = 5
        # perturb_side = 0

        start_time = time.time()
        score_cache = {}
        ils.score_cache = ils.read_update_write(score_cache, DIR_OUTPUT)

        # 初始化序列
        homebase_seq = init_sent.split(" ")

        homebase_miniseq = ils.seq2miniseq(homebase_seq)
        # print('warming up......')
        # miniseqs = [homebase_miniseq] + [random.sample(homebase_miniseq, len(homebase_miniseq)) for i in range(200)]
        globe_bst = [ils.cal_ppl_miniseqs([homebase_miniseq], bs=1)[0], homebase_miniseq]

        for n_search in range(1, 501):
            print("#" * 40, n_search, "#" * 40)
            miniseq = perturb(homebase_miniseq, perturb_rank)

            cur_miniseq, cur_score = ils.repeat_sa_search(
                miniseq=miniseq,
                globe_bst=globe_bst,
                n_search=n_search,
                temp_start=temp_start,
                temp_end=temp_end,
                cooling_rate=cooling_rate,
                max_epoch=max_epoch,
                patience=patience,
                steps_per_temp_factor=steps_per_epoch,
                bs_mul=bs_mul,
                verbose=verbose
            )

            if cur_score < globe_bst[0]:
                # perturb_side = 0
                perturb_rank = perturb_rank - max(1, int(perturb_rank/3))
                perturb_rank = max(perturb_rank,1)
                globe_bst[0] = cur_score
                globe_bst[1] = cur_miniseq
                homebase_miniseq = cur_miniseq.copy()
                save_top_100_keys_to_csv(ils.score_cache,
                                         f"{DIR_OUTPUT}/top_100_items.{ramdom_tag}.csv", ils)
            elif cur_score < pre_score:
                print('cur_score < pre_score!')
                perturb_rank = perturb_rank - max(1, int(perturb_rank/6))
                perturb_rank = max(perturb_rank,1)
                homebase_miniseq = cur_miniseq.copy()
            else:
                print('cur_score > pre_score')
                perturb_rank = 35
                homebase_miniseq = globe_bst[1].copy()

            pre_score = cur_score
            # elif shuffle_length >= ils.N//3:
            #     perturb_rank = 0
            #     perturb_side = (perturb_side + 1) % 3
            #     if perturb_side == 0:
            #         print("\nRandomly move homebase!!!")
            #         homebase_miniseq, _ = perturb(homebase_miniseq, 20, perturb_side)

            # ils.score_cache = ils.read_update_write(ils.score_cache, DIR_OUTPUT)
            hours, minutes, seconds = human_time(start_time)
            print('avg_dist<=31:',(ils.distance_matrix<=31).sum(),'!!!!')
            print(f"Running time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

        # 发送退出信号
        task_flag = torch.tensor(-1, device=DEVICE)
        dist.broadcast(task_flag, src=0)
    else:
        while True:
            # 从节点等待任务标志
            task_flag = torch.tensor(0, device=DEVICE)
            dist.broadcast(task_flag, src=0)

            if task_flag.item() == 1:
                ils.multi_gpu_infer(verbose=verbose)

            if task_flag.item() == -1:
                print(f"[Rank {dist.get_rank()}] Received exit signal, exiting...")
                break

    # 清理分布式环境
    dist.destroy_process_group()


if __name__ == '__main__':
    run_one_exp()



