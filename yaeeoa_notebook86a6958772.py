import json, time
import orjson
from collections import Counter
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# 1) Параметры и пути
DATA_DIR       = '/kaggle/input/otto-recommender-system'
TRAIN_PATH     = f'{DATA_DIR}/train.jsonl'
TEST_PATH      = f'{DATA_DIR}/test.jsonl'
MAX_TRAIN_SESS = 2_000_000
TOP_K_POP      = 100
WEIGHTS        = {'clicks':1,'carts':3,'orders':5}

# 2) Собираем глобальную популярность (пул кандидатов)
pop = Counter()
t0 = time.time()
with open(TRAIN_PATH,'r') as f:
    for i, line in enumerate(f,1):
        sess = orjson.loads(line)
        for ev in sess['events']:
            pop[ev['aid']] += WEIGHTS[ev['type']]
        if i >= MAX_TRAIN_SESS:
            break
print(f"[pop] {i} сессий за {time.time()-t0:.1f}s")
top_pop = [aid for aid,_ in pop.most_common(TOP_K_POP)]

# 3) Формируем обучающую выборку с фичами и метками
rows, t0 = [], time.time()
with open(TRAIN_PATH,'r') as f:
    for i, line in enumerate(f,1):
        if i > MAX_TRAIN_SESS: break
        evs = orjson.loads(line)['events']
        # последние 5 уникальных aid
        last5 = []
        for ev in reversed(evs):
            a = ev['aid']
            if a not in last5: last5.append(a)
            if len(last5)==5: break
        # кандидаты = last5 + дополняем популярными
        cands = list(last5)
        for a in top_pop:
            if len(cands)>=10: break
            if a not in cands: cands.append(a)
        # инициализируем stats с правильными ключами
        stats = {a:{'cnt_clicks':0,'cnt_carts':0,'cnt_orders':0,
                    'first_pos':-1,'last_pos':-1} for a in cands}
        for idx, ev in enumerate(evs):
            a,t = ev['aid'], ev['type']
            if a in stats:
                s = stats[a]
                if s['first_pos']==-1: s['first_pos']=idx
                s['last_pos']=idx
                s[f'cnt_{t}'] += 1
        # записываем строки
        for a,s in stats.items():
            rows.append({
                'cnt_clicks': s['cnt_clicks'],
                'cnt_carts':  s['cnt_carts'],
                'cnt_orders': s['cnt_orders'],
                'first_pos':  s['first_pos'],
                'last_pos':   s['last_pos'],
                'pop':        pop[a],
                'label':      int(s['cnt_orders']>0)
            })
print(f"[train df] {len(rows)} строк за {time.time()-t0:.1f}s")
df = pd.DataFrame(rows)
features = ['cnt_clicks','cnt_carts','cnt_orders','first_pos','last_pos','pop']

# 4) Обучаем LGBMClassifier
X, y = df[features], df['label']
tr, vl = train_test_split(df, test_size=0.1, random_state=42, stratify=y)
model = LGBMClassifier(objective='binary', n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42)
t1 = time.time()
model.fit(tr[features], tr['label'])
print(f"[lgbm] обучено за {time.time()-t1:.1f}s")

# 5) Предсказание и формирование submission_two_level.csv
out, t2 = [], time.time()
with open(TEST_PATH,'r') as f:
    for line in f:
        sess = orjson.loads(line)
        evs  = sess['events']
        # recency
        last5 = []
        for ev in reversed(evs):
            a = ev['aid']
            if a not in last5: last5.append(a)
            if len(last5)==5: break
        # кандидаты
        cands = list(last5)
        for a in top_pop:
            if len(cands)>=10: break
            if a not in cands: cands.append(a)
        # фичи теста
        feats = []
        for a in cands:
            s = {'cnt_clicks':0,'cnt_carts':0,'cnt_orders':0,
                 'first_pos':-1,'last_pos':-1}
            for idx, ev in enumerate(evs):
                if ev['aid']==a:
                    if s['first_pos']==-1: s['first_pos']=idx
                    s['last_pos']=idx
                    s[f'cnt_{ev["type"]}'] += 1
            feats.append({**s, 'pop': pop[a], 'aid': a})
        tmp = pd.DataFrame(feats)
        tmp['score'] = model.predict_proba(tmp[features])[:,1]
        top10 = tmp.sort_values('score', ascending=False).head(10)['aid'].tolist()
        lab = ' '.join(map(str, top10))
        sid = sess['session']
        out += [
            {'session_type':f'{sid}_clicks','labels':lab},
            {'session_type':f'{sid}_carts', 'labels':lab},
            {'session_type':f'{sid}_orders','labels':lab}
        ]
print(f"[pred] сформировано {len(out)} строк за {time.time()-t2:.1f}s")
pd.DataFrame(out).to_csv('submission_two_level.csv', index=False)

