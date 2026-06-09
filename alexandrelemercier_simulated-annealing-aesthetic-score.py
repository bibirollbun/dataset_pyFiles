!pip install -U bitsandbytes
!pip install git+https://github.com/openai/CLIP.git -q
!pip install cairosvg -q


import io
import random
import xml.etree.ElementTree as ET
from math import exp
from tqdm.notebook import tqdm  # or just `tqdm` if not in a notebook
from PIL import Image
import torch
import kagglehub
import torch.nn as nn
import clip
import gc
import cairosvg
from statistics import mean
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

BLANK_START = True

############################
# 1) The VQA Evaluator (LB score)
############################

class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description."""

    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Download or load the PaliGemma model locally
        self.model_path = kagglehub.model_download(
            'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        )
        self.questions = {
            'fidelity': 'Does <image> portray "{}" without any lettering? Answer yes or no.',
            'text': '<image> Text present: yes or no?',
        }

    def score(self, image: Image.Image, description: str) -> float:
        """
        Evaluates the fidelity of 'image' to the target 'description' using yes/no probabilities:
          - p_fidelity: Probability it matches the description
          - p_text: Probability there's text
        The final LB-like score is p_fidelity * (1 - p_text).
        """
        p_fidelity = self.get_yes_probability(image, self.questions['fidelity'].format(description))
        p_text = self.get_yes_probability(image, self.questions['text'])
        return p_fidelity * (1 - p_text)

    def mask_yes_no(self, logits):
        """Masks logits for 'yes' or 'no'."""
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        mask = torch.full_like(logits, float('-inf'))
        mask[:, yes_token_id] = logits[:, yes_token_id]
        mask[:, no_token_id] = logits[:, no_token_id]
        mask[:, yes_with_space_token_id] = logits[:, yes_with_space_token_id]
        mask[:, no_with_space_token_id] = logits[:, no_with_space_token_id]
        return mask

    def get_yes_probability(self, image, prompt) -> float:
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last token
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)

        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        prob_yes = probabilities[0, yes_token_id].item()
        prob_no = probabilities[0, no_token_id].item()
        prob_yes_space = probabilities[0, yes_with_space_token_id].item()
        prob_no_space = probabilities[0, no_with_space_token_id].item()

        total_yes_prob = prob_yes + prob_yes_space
        total_no_prob = prob_no + prob_no_space

        total_prob = total_yes_prob + total_no_prob
        renormalized_yes_prob = total_yes_prob / total_prob if total_prob > 0 else 0.0

        return renormalized_yes_prob


############################
# 2) The Aesthetic Evaluator
############################

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model_path = '/kaggle/input/sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = '/kaggle/input/openai-clip-vit-large-patch14/ViT-L-14.pt'
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:1')
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:1')
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device='cuda:1')
        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image, scaled into [0,1]."""
        image = self.preprocessor(image).unsqueeze(0).to('cuda:1')
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())
        return score.item() / 10.0  # scale to [0, 1]


############################
# 3) Utility: Convert SVG -> PNG
############################

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """Converts an SVG string to a PNG image using CairoSVG."""
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


############################
# 4) Mutation function
############################

def random_color() -> str:
    """Returns a random color string in 'rgb(r, g, b)' format."""
    return f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'


def mutate_svg(svg_text: str) -> str:
    """
    Mutates the SVG by either removing a random top-level element
    or adding a random shape (rect, circle, line, polygon).
    """
    tree = ET.ElementTree(ET.fromstring(svg_text))
    root = tree.getroot()

    # Decide how to mutate
    if len(svg_text.encode('utf-8')) > 9900:
        # If nearly at size limit, forcibly remove something
        if len(root) > 0:
            to_remove = random.choice(list(root))
            root.remove(to_remove)
    elif random.random() < 0.5 and len(root) > 0:
        # 50% chance to remove something
        to_remove = random.choice(list(root))
        root.remove(to_remove)
    else:
        # Otherwise add a new shape
        element_type = random.choice(['rect', 'circle', 'line', 'polygon'])

        if element_type == 'rect':
            elem = ET.Element('rect')
            elem.set('x', str(random.randint(0, 384)))
            elem.set('y', str(random.randint(0, 384)))
            elem.set('width', str(random.randint(10, 100)))
            elem.set('height', str(random.randint(10, 100)))
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        elif element_type == 'circle':
            elem = ET.Element('circle')
            elem.set('cx', str(random.randint(0, 384)))
            elem.set('cy', str(random.randint(0, 384)))
            elem.set('r', str(random.randint(5, 50)))
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        elif element_type == 'line':
            elem = ET.Element('line')
            elem.set('x1', str(random.randint(0, 384)))
            elem.set('y1', str(random.randint(0, 384)))
            elem.set('x2', str(random.randint(0, 384)))
            elem.set('y2', str(random.randint(0, 384)))
            elem.set('stroke', random_color())
            elem.set('stroke-width', str(random.randint(1, 5)))

        elif element_type == 'polygon':
            elem = ET.Element('polygon')
            points = " ".join(
                f"{random.randint(0,384)},{random.randint(0,384)}"
                for _ in range(random.randint(3, 6))
            )
            elem.set('points', points)
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        root.append(elem)

    return ET.tostring(root, encoding='unicode')


############################
# 5) Simulated Annealing with both Aesthetic + LB scoring
############################

# Initialize your evaluators
aesthetic_evaluator = AestheticEvaluator()
vqa_evaluator = VQAEvaluator()

# Example set of 20 scenes
scenes = [
    "A goose wearing a gold medal",
    "A lonely robot in a desert with red sand",
    "Surreal floating islands in a pink sky",
    "A medieval castle on top of a hill at night",
    "A futuristic cityscape at dusk with neon lights",
    "A close-up of a ladybug on a leaf covered with dew",
    "A magical library with flying books",
    "A giant coffee cup as a house in a rural landscape",
    "A portrait of a cat wearing a Victorian dress",
    "An underwater city with glass tunnels and fish",
    "A lion playing a guitar",
    "A turtle in a top hat reading a newspaper",
    "A superhero banana fighting crime",
    "An elephant painting a self-portrait",
    "A dragon sleeping in a child's bedroom",
    "A steampunk submarine in a sky ocean",
    "A chocolate skyscraper in a candy city",
    "A snowman on the surface of Mars",
    "A turtle cooking pizza in a fancy kitchen",
    "A neon-lit forest with glowing mushrooms",
]

# 1) Initialize current solution
if BLANK_START:
    current_svg = '<svg width="384" height="384" viewBox="0 0 384 384"></svg>'
else:
    current_svg = """<svg width="384" height="384" viewBox="0 0 384 384">
<rect x="99" y="172" width="17" height="97" fill="rgb(127, 173, 172)" stroke="black" stroke-width="1" />
<line x1="12" y1="165" x2="373" y2="275" stroke="rgb(160, 170, 82)" stroke-width="3" />
<rect x="161" y="226" width="29" height="50" fill="rgb(251, 56, 64)" stroke="black" stroke-width="1" />
<polygon points="102,254 283,59 8,180 379,117 182,209"
         fill="rgb(102, 179, 93)" stroke="black" stroke-width="1" />
<polygon points="371,114 242,256 112,212"
         fill="rgb(221, 242, 57)" stroke="black" stroke-width="1" />
<rect x="24" y="28" width="38" height="46" fill="rgb(131, 35, 141)" stroke="black" stroke-width="1" />
<polygon points="307,41 265,207 366,133 62,260 260,304"
         fill="rgb(82, 95, 12)" stroke="black" stroke-width="1" />
<circle cx="282" cy="176" r="14" fill="rgb(184, 157, 75)" stroke="black" stroke-width="1" />
<polygon points="122,32 114,285 38,336 13,314 150,289"
         fill="rgb(87, 157, 30)" stroke="black" stroke-width="1" />
<line x1="36" y1="350" x2="193" y2="121" stroke="rgb(197, 45, 71)" stroke-width="2" />
<line x1="109" y1="101" x2="307" y2="249" stroke="rgb(203, 185, 98)" stroke-width="1" />
<rect x="25" y="90" width="70" height="21" fill="rgb(95, 143, 217)" stroke="black" stroke-width="1" />
<line x1="281" y1="29" x2="152" y2="190" stroke="rgb(54, 164, 244)" stroke-width="5" />
<circle cx="341" cy="55" r="20" fill="rgb(246, 105, 21)" stroke="black" stroke-width="1" />
<rect x="265" y="275" width="69" height="71" fill="rgb(90, 30, 144)" stroke="black" stroke-width="1" />
<circle cx="230" cy="47" r="31" fill="rgb(216, 169, 126)" stroke="black" stroke-width="1" />
<line x1="215" y1="250" x2="165" y2="363" stroke="rgb(65, 210, 121)" stroke-width="4" />
<line x1="218" y1="320" x2="369" y2="350" stroke="rgb(94, 242, 133)" stroke-width="2" />
<line x1="177" y1="77" x2="234" y2="63" stroke="rgb(152, 229, 237)" stroke-width="2" />
<line x1="212" y1="96" x2="294" y2="97" stroke="rgb(35, 82, 183)" stroke-width="1" />
<line x1="179" y1="58" x2="295" y2="107" stroke="rgb(46, 219, 66)" stroke-width="1" />
<line x1="303" y1="54" x2="149" y2="172" stroke="rgb(228, 190, 62)" stroke-width="2" />
<polygon points="229,236 240,274 269,288 215,281"
         fill="rgb(99, 191, 27)" stroke="black" stroke-width="1" />
<circle cx="282" cy="117" r="13" fill="rgb(161, 136, 24)" stroke="black" stroke-width="1" />
<line x1="188" y1="296" x2="93" y2="384" stroke="rgb(25, 145, 106)" stroke-width="5" />
<line x1="183" y1="32" x2="247" y2="111" stroke="rgb(130, 226, 217)" stroke-width="1" />
<rect x="29" y="335" width="26" height="23" fill="rgb(131, 64, 228)" stroke="black" stroke-width="1" />
<rect x="87" y="263" width="66" height="57" fill="rgb(150, 126, 33)" stroke="black" stroke-width="1" />
<rect x="122" y="301" width="25" height="15" fill="rgb(208, 180, 164)" stroke="black" stroke-width="1" />
<line x1="359" y1="343" x2="284" y2="210" stroke="rgb(68, 251, 167)" stroke-width="3" />
<rect x="293" y="179" width="93" height="29" fill="rgb(183, 45, 241)" stroke="black" stroke-width="1" />
<rect x="300" y="102" width="11" height="12" fill="rgb(17, 243, 201)" stroke="black" stroke-width="1" />
<circle cx="359" cy="239" r="8" fill="rgb(123, 98, 179)" stroke="black" stroke-width="1" />
<line x1="157" y1="247" x2="229" y2="288" stroke="rgb(130, 149, 60)" stroke-width="1" />
<line x1="13" y1="326" x2="303" y2="333" stroke="rgb(53, 96, 95)" stroke-width="2" />
<polygon points="240,181 273,276 271,326 272,279"
         fill="rgb(76, 240, 5)" stroke="black" stroke-width="1" />
<rect x="188" y="259" width="39" height="52" fill="rgb(203, 217, 25)" stroke="black" stroke-width="1" />
<circle cx="154" cy="198" r="8" fill="rgb(7, 89, 140)" stroke="black" stroke-width="1" />
<circle cx="202" cy="141" r="10" fill="rgb(4, 236, 142)" stroke="black" stroke-width="1" />
<line x1="347" y1="126" x2="200" y2="160" stroke="rgb(215, 228, 118)" stroke-width="1" />
<rect x="228" y="289" width="27" height="30" fill="rgb(83, 126, 71)" stroke="black" stroke-width="1" />
<rect x="326" y="358" width="100" height="61" fill="rgb(103, 181, 227)" stroke="black" stroke-width="1" />
<line x1="3" y1="376" x2="97" y2="206" stroke="rgb(148, 83, 204)" stroke-width="4" />
<rect x="361" y="359" width="80" height="74" fill="rgb(103, 100, 82)" stroke="black" stroke-width="1" />
<circle cx="148" cy="218" r="10" fill="rgb(2, 241, 243)" stroke="black" stroke-width="1" />
<line x1="179" y1="165" x2="1" y2="374" stroke="rgb(24, 174, 111)" stroke-width="2" />
<line x1="219" y1="256" x2="376" y2="309" stroke="rgb(121, 204, 19)" stroke-width="5" />
<circle cx="309" cy="83" r="5" fill="rgb(231, 230, 175)" stroke="black" stroke-width="1" />
<line x1="312" y1="145" x2="253" y2="142" stroke="rgb(179, 186, 105)" stroke-width="2" />
<line x1="185" y1="309" x2="186" y2="294" stroke="rgb(30, 49, 143)" stroke-width="2" />
<line x1="125" y1="233" x2="213" y2="199" stroke="rgb(58, 60, 114)" stroke-width="1" />
<line x1="267" y1="203" x2="271" y2="283" stroke="rgb(165, 70, 115)" stroke-width="2" />
<line x1="258" y1="77" x2="136" y2="124" stroke="rgb(13, 94, 63)" stroke-width="2" />
<line x1="147" y1="218" x2="76" y2="27" stroke="rgb(132, 221, 5)" stroke-width="4" />
<line x1="131" y1="124" x2="196" y2="123" stroke="rgb(82, 7, 68)" stroke-width="1" />
<line x1="70" y1="354" x2="99" y2="321" stroke="rgb(94, 99, 35)" stroke-width="3" />
<rect x="348" y="351" width="73" height="68" fill="rgb(159, 84, 187)" stroke="black" stroke-width="1" />
<line x1="231" y1="346" x2="107" y2="245" stroke="rgb(85, 159, 38)" stroke-width="2" />
<line x1="148" y1="284" x2="27" y2="188" stroke="rgb(190, 195, 106)" stroke-width="4" />
<line x1="368" y1="122" x2="314" y2="190" stroke="rgb(104, 183, 193)" stroke-width="3" />
<line x1="146" y1="264" x2="0" y2="115" stroke="rgb(127, 224, 109)" stroke-width="2" />
<circle cx="161" cy="175" r="7" fill="rgb(95, 88, 16)" stroke="black" stroke-width="1" />
<rect x="333" y="65" width="97" height="16" fill="rgb(157, 135, 236)" stroke="black" stroke-width="1" />
<line x1="97" y1="146" x2="204" y2="216" stroke="rgb(20, 61, 101)" stroke-width="2" />
<circle cx="184" cy="88" r="8" fill="rgb(135, 58, 176)" stroke="black" stroke-width="1" />
<line x1="188" y1="55" x2="211" y2="88" stroke="rgb(92, 122, 143)" stroke-width="4" />
<rect x="242" y="270" width="28" height="49" fill="rgb(232, 223, 32)" stroke="black" stroke-width="1" />
<line x1="260" y1="60" x2="276" y2="59" stroke="rgb(40, 101, 26)" stroke-width="4" />
<circle cx="73" cy="309" r="8" fill="rgb(17, 151, 26)" stroke="black" stroke-width="1" />
<circle cx="287" cy="225" r="6" fill="rgb(59, 160, 53)" stroke="black" stroke-width="1" />
<line x1="68" y1="292" x2="112" y2="239" stroke="rgb(250, 187, 77)" stroke-width="4" />
<rect x="138" y="362" width="27" height="15" fill="rgb(7, 129, 169)" stroke="black" stroke-width="1" />
<circle cx="167" cy="172" r="5" fill="rgb(126, 71, 21)" stroke="black" stroke-width="1" />
<line x1="345" y1="339" x2="346" y2="323" stroke="rgb(105, 160, 158)" stroke-width="4" />
<line x1="27" y1="330" x2="237" y2="315" stroke="rgb(21, 72, 85)" stroke-width="1" />
<rect x="280" y="14" width="16" height="36" fill="rgb(192, 77, 222)" stroke="black" stroke-width="1" />
<polygon points="86,208 111,205 63,245"
         fill="rgb(255, 226, 132)" stroke="black" stroke-width="1" />
<line x1="25" y1="328" x2="50" y2="329" stroke="rgb(101, 74, 46)" stroke-width="4" />
<line x1="84" y1="337" x2="82" y2="311" stroke="rgb(153, 133, 32)" stroke-width="1" />
<line x1="238" y1="309" x2="158" y2="285" stroke="rgb(57, 14, 34)" stroke-width="2" />
<line x1="34" y1="168" x2="67" y2="198" stroke="rgb(156, 11, 215)" stroke-width="2" />
<line x1="183" y1="311" x2="211" y2="306" stroke="rgb(205, 36, 229)" stroke-width="1" />
<line x1="356" y1="328" x2="296" y2="266" stroke="rgb(51, 159, 70)" stroke-width="1" />
<circle cx="173" cy="191" r="6" fill="rgb(58, 215, 20)" stroke="black" stroke-width="1" />
<line x1="357" y1="320" x2="289" y2="342" stroke="rgb(75, 223, 3)" stroke-width="3" />
<circle cx="147" cy="331" r="5" fill="rgb(63, 14, 31)" stroke="black" stroke-width="1" />
<line x1="180" y1="306" x2="137" y2="343" stroke="rgb(118, 74, 181)" stroke-width="3" />
<circle cx="89" cy="264" r="5" fill="rgb(74, 199, 26)" stroke="black" stroke-width="1" />
<line x1="152" y1="320" x2="145" y2="320" stroke="rgb(56, 245, 168)" stroke-width="2" />
<circle cx="174" cy="308" r="5" fill="rgb(38, 226, 229)" stroke="black" stroke-width="1" />
<line x1="245" y1="145" x2="189" y2="165" stroke="rgb(52, 108, 131)" stroke-width="1" />
<line x1="316" y1="165" x2="298" y2="168" stroke="rgb(133, 107, 100)" stroke-width="4" />
<line x1="66" y1="282" x2="93" y2="245" stroke="rgb(65, 165, 50)" stroke-width="4" />
</svg>"""
current_img = svg_to_png(current_svg)
current_aesthetic = aesthetic_evaluator.score(current_img)
current_img


def compute_mean_lb(img: Image.Image, vqa: VQAEvaluator, scenes: list, n=10) -> float:
    """Compute mean LB score across 'n' random scenes."""
    total = 0.0
    for _ in range(n):
        sc = random.choice(scenes)
        total += vqa.score(img, sc)
    return total / n

# Choose random scene for LB check
scene = random.choice(scenes)
current_lb = vqa_evaluator.score(current_img, scene)

# 2) Initialize global best
best_svg = current_svg
best_aesthetic = current_aesthetic
best_lb = current_lb

# 3) Simulated Annealing parameters
max_iter = 100_000
initial_temp = 0.15
cooling_rate = 0.9993
temperature = initial_temp

# We'll only do LB evaluation "100 times" after iteration >= 100
lb_evals_done = 0
LB_EVALS_MAX = 2

best_lb = 0.0
current_lb = 0.0
best_aes = 0.0
current_aes = 0.0

history = []

pbar = tqdm(range(max_iter), desc="Simulated Annealing (Aesthetic)")

for i in pbar:
    gc.collect()
    # 1) Mutate
    next_svg = mutate_svg(current_svg)
    next_img = svg_to_png(next_svg)
    next_aes = aesthetic_evaluator.score(next_img)

    # 2) Accept or not, based on aesthetic
    delta = next_aes - current_aes
    if delta > 0:
        # better
        current_svg = next_svg
        current_aes = next_aes
    else:
        # maybe accept
        if random.random() < exp(delta / temperature):
            current_svg = next_svg
            current_aes = next_aes

    # 3) Update best if needed
    if current_aes > best_aes:
        best_svg = current_svg
        best_aes = current_aes
        history.append(best_aes)

    # 4) Sometimes compute LB
    # Only if i >= 100 and haven't done all 100 LB checks:
    if i >= 100 and lb_evals_done < LB_EVALS_MAX:
        # Compute mean LB across 10 random scenes
        mean_lb = compute_mean_lb(svg_to_png(current_svg), vqa_evaluator, scenes, n=10)
        current_lb = mean_lb

        # Update best LB if it's higher
        if current_lb > best_lb:
            best_lb = current_lb
        lb_evals_done += 1

    # 5) Cool down
    temperature *= cooling_rate

    # 6) Show progress
    pbar.set_postfix_str(
        f"T={temperature:.4f}, bestAes={best_aes:.3f}, currAes={current_aes:.3f}, bestLB={best_lb:.3f}, currLB={current_lb:.3f}, lbEvals={lb_evals_done}"
    )

print("Done!")
print(f"Best aesthetic found: {best_aes:.3f}")
print(f"Best LB (observed):   {best_lb:.3f}")
# 'best_svg' is your final aesthetic-optimized solution


len(best_svg.encode('utf-8'))


best_svg


image = svg_to_png(best_svg)
image


import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 13

plt.plot(history)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("best_aesthetic_score")

