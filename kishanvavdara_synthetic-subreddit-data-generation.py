!pip install -q aiolimiter


%%writefile script.py


import pandas as pd 
import os, csv, asyncio, itertools, logging, time
from pathlib import Path
import pandas as pd
from aiolimiter import AsyncLimiter
from rich.progress import Progress, BarColumn, TimeElapsedColumn
from openai import AsyncOpenAI
import logging
from kaggle_secrets import UserSecretsClient


user_secrets = UserSecretsClient()
OPENROUTER_API_KEY = user_secrets.get_secret("OPENROUTER_API_KEY")


class SuppressHTTPRequests(logging.Filter):
    def filter(self, rec):
        return "HTTP Request:" not in rec.getMessage()


logging.getLogger().addFilter(SuppressHTTPRequests())

# â•­â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â•®
N_POS, N_NEG      = 3, 3
CHECKPOINT_EVERY  = 500
MAX_GEN_LEN       = 220
PREVIEW_ROWS      = 10


# Models for generating positive samples
POS_MODELS = itertools.cycle([
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-4-scout"])

# Models for generating positive samples
NEG_MODELS = itertools.cycle([
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mixtral-8x7b-instruct",
])

OUT_DIR = Path("generate_synth_data/synth_ckpt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_LIM = AsyncLimiter(120, 60)        # 120 generator calls / min
# â•°â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â•¯


assert OPENROUTER_API_KEY, "export OPENROUTER_API_KEY first"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY)

logging.basicConfig(
    filename="synth_gen_1.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s")


# â€“â€“â€“ prompt templates â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“
SYS_GEN = (
    "You are a long-time Redditor who can adopt ANY voiceâ€”from wholesome dad-jokes "
    "to edgy NSFW roasts.\n\n"
    "â”‚  Subreddit blurb  â”‚\n"
    "{desc}\n\n"

    "â–¸ Write **{n} standalone Reddit comments** (â‰¤ 20 words each).\n"
    "   â€¢   *Each comment must* **{sense}** the rule shown below *exactly as written*:\n"
    "       Â«{rule}Â»\n\n"

    "â–¸ Style / content variety\n"
    "   Pick **at least one** random twist for *each* comment from the pool â†“. "
    "   **Do not reuse the same twist pattern on consecutive lines**.\n"
    "   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€\n"
    "   â‘  address OP directly (e.g. â€œOP, â€¦â€� or â€œu/â€¦â€�) ğŸ‘¤\n"
    "   â‘¡ begin with a quoted fragment â€œ> â€¦â€� ğŸ—£ï¸�\n"
    "   â‘¢ drop a real-looking URL or .com link ğŸŒ�\n"
    "   â‘£ include strong / NSFW language ğŸ˜ˆ\n"
    "   â‘¤ sprinkle emoji or RANDOM CAPITALS ğŸ¤ª\n"
    "   â‘¥ be cryptic / absurd / off-topic ğŸ¤¯\n"
    "   â‘¦ pose a rhetorical question â�“\n"
    "   â‘§ use a hashtag or fake markdown #lifehack **bold** _italics_\n"
    "   â‘¨ reference a meme or pop-culture line (â€œYeet!â€�, â€œThis is fine.â€�) ğŸ�¶ğŸ”¥\n"
    "   â‘© extremely short reply (â€œNah.â€� / â€œ+1â€�) â€” but NOT fewer than 4 words\n"
    "   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€\n\n"

    "â–¸ Output rules\n"
    "   â€¢ produce **exactly {n} lines**, newline-separated.\n"
    "   â€¢ **NO** numbering, bullets, explanations or code fences.\n"
)

# â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“


async def chat(model: str, messages: list[dict]):
    async with GEN_LIM:
        try:
            rsp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_GEN_LEN,
                temperature=1.5)
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            logging.error("model %s failed â€“ %s", model, e)
            return ""


async def generate(rule: str, desc: str, positive: bool) -> list[tuple[str,str]]:
    n   = N_POS if positive else N_NEG
    mdl = next(POS_MODELS) if positive else next(NEG_MODELS)

    sys_msg = SYS_GEN.format(
        desc=desc or "No description",
        n=n,
        sense="BREAKS" if positive else "COMPLIES with",
        rule=rule,
    )
    txt = await chat(mdl,
        [{"role": "system", "content": sys_msg},
         {"role": "user",   "content": "Begin"}],
    )

    unique, seen = [], set()
    for ln in txt.splitlines():
        s = ln.strip()
        if s and s.lower() not in seen:
            unique.append((s, mdl))
            seen.add(s.lower())
    return unique[:n]


async def handle_rule(row, writer, task_id, prog):
    """Generate for one rule row, write accepted lines, update bar."""
    start = time.time()
    sub, rule, desc = row.subreddit, row.rule_prompt, row.sub_description

    # positives then negatives
    for positive in (True, False):
        lbl  = 1 if positive else 0
        lines = await generate(rule, desc, positive)
        for body, gen_mdl in lines:
            writer.writerow(dict(
                subreddit=sub,
                sub_description=desc,
                rule_prompt=rule,
                body=body,
                label=lbl,
                synthetic=1,
                gen_model=gen_mdl,
            ))
            prog.update(task_id, advance=1)

    dt = time.time() - start
    logging.info("rule %s done in %.1fs", sub, dt)


# â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“ main â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“â€“
async def main():
    rules = pd.read_csv("/kaggle/input/subreddit-rules-dataset/reddit_rules_dataset_public_3krules.csv")
    # Use only 10 rows
    rules = rules[:10]
    rules.reset_index(inplace=True, drop=True)
    total = len(rules) * (N_POS + N_NEG)
    print("Loaded data starting generation...")
    ckpt_idx, written = 0, 0

    def new_writer(i: int):
        fp = OUT_DIR / f"synth_fast_{i:03d}.csv"
        f = fp.open("w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=[
            "subreddit", "sub_description", "rule_prompt", "body",
            "label", "synthetic", "gen_model"])
        w.writeheader()
        return f, w, fp

    f, writer, first_file = new_writer(ckpt_idx)
    preview_rows = None

    with Progress("[progress.description]{task.description}",
                  BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeElapsedColumn()) as prog:
        t_id = prog.add_task("Generating", total=total)

        for row in rules.itertuples():
            try:
                await handle_rule(row, writer, t_id, prog)
            except Exception as e:
                logging.error("Failure on %s â€“ %s", row.subreddit, e)

            written += N_POS + N_NEG
            if written >= CHECKPOINT_EVERY:
                f.flush(); f.close()
                if preview_rows is None:
                    preview_rows = pd.read_csv(first_file).head(PREVIEW_ROWS)
                ckpt_idx += 1
                f, writer, first_file = new_writer(ckpt_idx)
                written = 0

    f.close()
    if preview_rows is None:
        preview_rows = pd.read_csv(first_file).head(PREVIEW_ROWS)

    print(f"\nFirst {PREVIEW_ROWS} rows:")
    print(preview_rows.to_markdown(index=False))
    preview_rows.to_csv("preview_first10.csv", index=False)
    print("â†³ Saved preview_first10.csv")


if __name__ == "__main__":
    asyncio.run(main())



import subprocess

result = subprocess.run(["python", "script.py"], capture_output=True, text=True)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)


import pandas as pd
import re
df = pd.read_csv("/kaggle/working/generate_synth_data/synth_ckpt/synth_fast_000.csv")

# clean some stuff
df['body'] = df['body'].str.replace(r'^\s*(>\s*)*(OP,?\s*)*', '', regex=True) 
df.head(10)

