import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasonOrIntuition(nn.Module):

    def __init__(self, confidence_level = 0.9, max_recursion = 3, temperature = 0.5):
        super().__init__()
        
        self.confidence_level = confidence_level
        self.max_recursion = max_recursion
        self.temperature = temperature
      
    def familiar(self,x):
        # f: X -> [0,1]
        # Note: returns a single value even for a batch
        return torch.rand(1)

    def intuition(self,x):
        # f: X -> Y
        return f"I({x})"

    def policy(self,x):
        # f[i]: X -> [0,1]^N
        # Note: returns a single vector even for a batch
        return torch.tensor([0.3,0.5,0.2])
        
    def neighbour(self,i,x):
        # f[i]: X -> 2^X, i = 1,...,N
        i = int(i)
        return [f"x{i}a",f"x{i}b"]

    def rule(self,i,Y,x):
        # f[i]: 2^Y x X -> Y, i = 1,...,N
        i = int(i)
        Y_str = "[" + ",".join(Y) + "]"
        return f"R{i}({Y_str},{x})"

    def ROI(self, x, recusion_level = 0):
        if self.familiar(x) >= self.confidence_level or recusion_level == self.max_recursion:
            return self.intuition(x)
        else:
            i = self.sample_policy(x)
            return self.reason(i, x, recusion_level)

    def reason(self, i, x, recusion_level = 0):
        X = self.neighbour(i,x)
        Y = [self.ROI(x_hat, recusion_level + 1) for x_hat in X]
        return self.rule(i, Y, x)
        
    def sample_policy(self, x):
        if self.temperature < 1e-3:
            return torch.argmax(self.policy(x))
        else:
            scaled = torch.softmax(self.policy(x)/self.temperature, dim=0)
            return torch.multinomial(scaled, 1)

    def __call__(self, x):
        return self.ROI(x)

    def think(self, x, task_loss, num_trials = 20, num_solutions = 1):
        # Note: if x is a batch, the entire batch follows the same policy.
        trial_solutions = []
        trial_losses = torch.empty(num_trials)
    
        for t in range(num_trials):
            y = self.ROI(x)
            trial_solutions.append(y)
            trial_losses[t] = task_loss(y)
    
        idx_best_solutions = torch.argsort(trial_losses)[:num_solutions]
        return [trial_solutions[i] for i in idx_best_solutions]


    def train_loss(self, X, Y, distance = F.mse_loss):
        L_tot = 0.0
        for x, y in zip(X, Y):
            x = x.unsqueeze(0)  # make it (1, D)
            y = y.unsqueeze(0)
    
            p = self.familiar(x)
            w = self.policy(x)
            w = w / torch.sum(w) # normalize
            N = w.numel()
            I = self.intuition(x)
            
            d_I = lambda z: distance(z,I)
            d_R = lambda z: torch.sum(torch.stack([w[i]*distance(z,self.reason(i,x)) for i in range(N)]))
            
            L_data = p*d_I(y) + (1-p)*d_R(y)
            L_think = (1-p)*d_R(I)
            L_tot += L_data + L_think      
        return L_tot
        
    def forward(self, x):
        return self.ROI(x)
    
    def train(self, X, Y, train_loss=None, epochs=10):
        if train_loss is None:
            train_loss = self.train_loss

        dataset = torch.utils.data.TensorDataset(X, Y)
        optimizer = torch.optim.Adam(self.parameters())        
        loader = torch.utils.data.DataLoader(dataset, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                loss = train_loss(X_batch, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")        


torch.manual_seed(0)
roi = ReasonOrIntuition(confidence_level=0.5)
loss = lambda y: torch.rand(1)
solutions = roi.think("x0", loss, num_solutions = 2)
print(solutions[0])
print(solutions[1])


import torch

class FactorialROI(ReasonOrIntuition):
    def __init__(self, max_recursion=20):
        super().__init__(confidence_level=0.9, max_recursion=max_recursion, temperature=0.0)

    def familiar(self, x):
        # Stop recursion when x <= 1
        return torch.tensor(1.0 if x <= 1 else 0.0)

    def intuition(self, x):
        # Base case: factorial(0 or 1) = 1
        return torch.tensor(1)

    def policy(self, x):
        # Only one possible recursive rule
        return torch.tensor([1.0])

    def neighbour(self, i, x):
        # Next recursive call: (x - 1)
        return [x - 1]

    def rule(self, i, Y, x):
        # Factorial rule: x * factorial(x - 1)
        return x * Y[0]


import math

roi = FactorialROI(max_recursion=6)

for n in range(1, 10):
    n_t = torch.tensor(n)
    roi_val = roi(n_t).item()
    true_val = math.factorial(n)
    ok = "✓" if roi_val == true_val else "✗"
    print(f"n={n:2d} ROI={roi_val:7d} true={true_val:7d} {ok}")


import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralStateROI(ReasonOrIntuition):
    def __init__(self, state_dim=1, num_actions=3, num_neurons=16):
        super().__init__()

        self.num_actions = num_actions

        self.familiar_net = nn.Sequential(
            nn.Linear(state_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 1),
            nn.Sigmoid()
        )

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_actions)
        )

        self.intuition_net = nn.Sequential(
            nn.Linear(state_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, state_dim)
        )

        self.next_state_net = nn.Sequential(
            nn.Linear(state_dim + num_actions, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, state_dim)
        )

    def familiar(self, x):
        out = self.familiar_net(x).squeeze(-1)
        return out.mean()  # always returns a single scalar (mean over batch)

    def policy(self, x):
        w = F.softmax(self.policy_net(x), dim=-1)  # (num_actions,) or (batch_size, num_actions)
        if w.dim() == 2:  # batch
            w = w.mean(dim=0)  # average over batch -> vector of num_actions
        w = w.flatten()       # ensure 1D vector
        return w

    def intuition(self, x):
        return self.intuition_net(x)

    def neighbour(self, i, x):
        if i == 0:
            return []
        else:            
            one_hot = torch.zeros(self.num_actions)
            one_hot[i] = 1.0
            
            if x.dim() == 1:
                xi = torch.cat([x, one_hot], dim=0)  # shape: [state_dim + num_actions]
            if x.dim() == 2:
                one_hot_batch = one_hot.unsqueeze(0).repeat(x.shape[0], 1)
                xi = torch.cat([x, one_hot_batch], dim=1)  # shape: [batch_size, state_dim + num_actions]
                
            return [self.next_state_net(xi)]

    def rule(self, i, Y, x):
        if i == 0:
            return x # stop action
        else:
            return Y[0] # final state from the next actions


import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

X = torch.randn(100, 4)
Y = 2 * X + 0.5 * torch.randn(100, 4)
model = NeuralStateROI(state_dim=4, num_actions=3)
model.train(X,Y, epochs=5)

task_loss = lambda y_pred: torch.mean(y_pred ** 2)
x_task = torch.randn(4)
model.temperature = 1
solutions = model.think(x_task, task_loss=task_loss, num_trials=2, num_solutions=2)

print("\nCandidate solutions:")
for i, sol in enumerate(solutions):
    print(f"Solution {i+1}: {sol}")


import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcROI(NeuralStateROI):
    def __init__(self, image_size=64, encoding_dim=8, num_actions=4, num_neurons=32):

        # state_dim includes the original image height and width (+2)
        state_dim = encoding_dim + 2
        super().__init__(state_dim = state_dim, num_actions=num_actions, num_neurons=num_neurons)

        self.image_size = image_size
        self.channels = 10 # ARC-AGI data

        self.encoder_net = nn.Sequential(
            nn.Conv2d(self.channels, 2*self.channels, kernel_size=3, stride=2, padding=1),  # halve spatial, double channels
            nn.ReLU(),
            nn.Conv2d(2*self.channels, 4*self.channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((image_size//4) * (image_size//4) * (4*self.channels), encoding_dim)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(encoding_dim, (image_size//4) * (image_size//4) * (4*self.channels)),
            nn.ReLU(),
            nn.Unflatten(1, (4*self.channels, image_size//4, image_size//4)),
            nn.ConvTranspose2d(4*self.channels, 2*self.channels, kernel_size=4, stride=2, padding=1),  # double spatial, halve channels
            nn.ReLU(),
            nn.ConvTranspose2d(2*self.channels, self.channels, kernel_size=4, stride=2, padding=1),
        )

    def vec2img(self, vec):
        # (B, C*H*W) -> (B, C, H, W)
        B = vec.shape[0]
        img = vec.view(B, self.channels, self.image_size, self.image_size)
        return img

    def img2vec(self, img):
        # (B, C, H, W) -> (B, C*H*W)
        B = img.shape[0]
        return img.view(B, -1)

    def preprocess(self, grids):
        """
        Takes a list of 2D grids (list of list of shape (H, W)),
        upscales to (image_size, image_size), one-hot encodes (10 channels),
        flattens, and prepends [H, W].
        Returns: (B, 2 + C*image_size*image_size)
        """
        batch = []
        for g in grids:
            g = torch.tensor(g,dtype=torch.float)
            H, W = g.shape
            one_hot = F.one_hot(g.long(), num_classes=self.channels).permute(2, 0, 1).float()  # (C,H,W)
            upscaled = F.interpolate(one_hot.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear')  # (1,C,H',W')
            flat = upscaled.flatten(start_dim=1)  # (1, C*H'*W')
            vec = torch.cat([torch.tensor([[H, W]], dtype=torch.float32), flat], dim=1)
            batch.append(vec)
        return torch.cat(batch, dim=0)  # (B, 2 + C*H'*W')

    def postprocess(self, vectors):
        """
        Given (B, 2 + C*H'*W'), reconstruct original grids:
        - Extract H,W
        - Reshape rest into (B, C, H', W')
        - Downscale to (H,W)
        - Argmax over channels → final grid
        """
        grids = []
        for v in vectors:
            H, W = v[:2].int().tolist()
            img_flat = v[2:]
            img = img_flat.view(self.channels, self.image_size, self.image_size).unsqueeze(0)
            downscaled = F.interpolate(img, size=(H, W), mode='bilinear')  # (1, C, H, W)
            grid = torch.argmax(downscaled, dim=1).squeeze(0).to(torch.int64)
            grids.append(grid.tolist())
        return grids
       
    def encode(self, x):
        # x: (B, 2 + C*H*W)
        img = self.vec2img(x[:, 2:])
        z_img = self.encoder_net(img)
        return torch.cat([x[:, :2], z_img], dim=1)

    def decode(self,z):
        # z: (B, 2 + encoding_dim)
        sizes = torch.clamp(z[:, :2], min=1, max=self.image_size)
        img = torch.sigmoid(self.decoder_net(z[:, 2:]))
        return torch.cat([sizes, self.img2vec(img)], dim=1)

    def familiar(self, x):
        return super().familiar(self.encode(x))

    def intuition(self, x):
        return self.decode(super().intuition(self.encode(x)))

    def policy(self, x):
        return super().policy(self.encode(x))

    def neighbour(self, i, x):
        return [self.decode(xj) for xj in super().neighbour(i, self.encode(x))]

    def rule(self, i, Y, x):
        return self.decode(super().rule(i, [self.encode(y) for y in Y], self.encode(x)))

    def train(self, X, Y, train_loss=None, epochs=10):
        if train_loss is None:
            train_loss = self.train_loss
        return super().train(self.preprocess(X), self.preprocess(Y), train_loss=train_loss, epochs=epochs)
    
    def predict(self, grids):
        return self.postprocess(self.forward(self.preprocess(grids)))

    def think(self, x_query, X, Y, num_trials=20, num_solutions=2):
        
        X_all_prep = self.preprocess([x_query] + X)        
        
        def task_loss(prediction):
            prediction = prediction[1:] # discard y_query for evaluation purposes
            pixel_loss = torch.mean(torch.abs((prediction - self.preprocess(Y)))) # each pixel value is in [0,1] ->  pixel_loss in [0,1]
            exact_match = torch.tensor([[p==y] for p, y in zip(self.postprocess(prediction), Y)],dtype=torch.float).mean()
            return -10*exact_match + pixel_loss
        
        Y_all = super().think(X_all_prep, task_loss, num_trials=num_trials, num_solutions=num_solutions)
        return [self.postprocess([y[0,:]]) for y in Y_all] # return only y_query


import json
with open("/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json") as f:
    arc_data = json.load(f)

X_grids, Y_grids = [], []
for task in arc_data.values():
    for pair in task["train"]:
        X_grids.append(pair["input"])
        Y_grids.append(pair["output"])

# Keep only the first n grids
n = 10
X_grids = X_grids[:n]
Y_grids = Y_grids[:n]

model = ArcROI(image_size=32, encoding_dim=8)
model.train(X_grids, Y_grids, epochs=5)


import json
import torch

# === Paths ===
test_path = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
output_path = "/kaggle/working/submission.json"

# === Solve a single task using ArcROI ===
def solve_task(task, model, num_solutions=2):
    X_train, Y_train = [], []
    for pair in task["train"]:
        X_train.append(pair["input"])
        Y_train.append(pair["output"])

    predictions = []
    for test in task["test"]:
        x_query = test["input"]
        solutions = model.think(x_query, X_train, Y_train, num_trials=5, num_solutions=num_solutions)
        pred_dict = {
            "attempt_1": solutions[0],
            "attempt_2": solutions[1]
        }
        predictions.append(pred_dict)
    return predictions

# === Load test set ===
with open(test_path, "r") as f:
    test_data = json.load(f)

# === Generate predictions ===
submission = {}
for task_id, task in test_data.items():
    submission[task_id] = solve_task(task, model, num_solutions=2)

# === Save to JSON ===
with open(output_path, "w") as f:
    json.dump(submission, f)

print(f"✅ Submission created with {len(submission)} tasks at {output_path}")

