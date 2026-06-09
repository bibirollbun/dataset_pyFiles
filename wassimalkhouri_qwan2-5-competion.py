#| default_exp core
# Drawing-with-LLMs â€“ inference-only Package

#| export
from transformers import AutoTokenizer, AutoModelForCausalLM
import kagglehub, pathlib, re, torch

ALLOWED = {"svg","rect","circle","ellipse","line","polyline","polygon",
           "path","text","g","defs"}

class Model:
    """Return an SVG string for a given text prompt."""

    def __init__(self):
        # Download the dataset -> e.g. /kaggle/input/qwen23-full
        root = pathlib.Path(
            kagglehub.dataset_download("wassimalkhouri/qwen23-full")
        )

        # ðŸ‘‰ merged model lives in the *sub-folder*:
        self.model_dir = root / "qwen25_vsp_full"

        # HF will now find tokenizer.json, model-00001-of-003 etc.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(self.model_dir,
                                       torch_dtype=torch.float16,
                                       device_map="auto")
                      .eval())

    # --- post-processing helpers ---
    def _clean(self, svg: str) -> str:
        svg = svg.split("<?xml")[-1].strip()
        if len(svg.encode()) > 10_000:
            svg = svg[:9_900] + "</svg>"
        bad = {m.group(2).lower()
               for m in re.finditer(r"<(/?)(\w+)", svg)
               if m.group(2).lower() not in ALLOWED}
        for tag in bad:
            svg = re.sub(fr"</?{tag}[^>]*>", "", svg, flags=re.I)
        return svg

    # --- competition entry point ---
    def predict(self, prompt: str) -> str:
        sys = ("You are Qwen. Respond **only** with valid SVG (â‰¤10 elements).")
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            msgs, return_tensors="pt").to("cuda")
        out = self.model.generate(ids, max_new_tokens=512, do_sample=False)
        svg = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._clean(svg)

#| hide
if __name__ == "__main__":
    import kaggle_evaluation as ke
    score = ke.test(Model)          # uses the tiny mock test.csv shipped in the competition repo
    print("Local SVG-Fidelity score on mock set:", score)
    model = Model()
    print(model.predict("Draw a circle"))

