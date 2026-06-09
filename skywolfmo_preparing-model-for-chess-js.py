!pip install chess


%%writefile nnue_pipeline.py
#!/usr/bin/env python3
# Requirements: pip install chess pandas numpy torch torchvision tqdm scikit-learn matplotlib seaborn

import argparse
import time
import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error


# ===============================
# Encoding utilities
# ===============================
pieces = list('rnbqkpRNBQKP.')  # 13 symbols
_piece_to_index = {p: i for i, p in enumerate(pieces)}

def one_hot_encode_piece(piece):
    arr = np.zeros(len(pieces), dtype=np.float32)
    arr[_piece_to_index[piece]] = 1.0
    return arr

def encode_board(board):
    board_str = str(board).replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        for piece in row:
            board_list.append(one_hot_encode_piece(piece))
    return np.array(board_list, dtype=np.float32)  # (64,13)

def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)  # (64,13)

INPUT_DIM = 64 * 13  # flattened board


def encode_series_to_np(series, desc="encode"):
    arrs = []
    for fen in tqdm(series, desc=desc):
        arrs.append(encode_fen_string(fen))
    return np.stack(arrs)  # (N,64,13)


# ===============================
# Models
# ===============================
class ChessAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class EvalNet(nn.Module):
    def __init__(self, autoenc, latent_dim=128):
        super().__init__()
        self.encoder = autoenc.encoder  # reuse pretrained encoder
        self.eval_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.eval_head(latent)
        return out


# ===============================
# Training helpers
# ===============================
def train_autoencoder(autoenc, train_loader, device, epochs=10, lr=1e-3):
    autoenc.to(device)
    optimizer = optim.Adam(autoenc.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        autoenc.train()
        total_loss = 0
        for xb, _ in train_loader:
            optimizer.zero_grad()
            recon, _ = autoenc(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Autoenc] Epoch {epoch}, Loss={avg_loss:.4f}")

    return autoenc


def train_evalnet(model, train_loader, val_loader, device, epochs=20, lr=2e-4):
    model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE

    best_val = float("inf")
    patience, bad_epochs = 5, 0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Validation
        val_loss = 0
        model.eval()
        if val_loader:
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        else:
            val_loss = float("nan")

        print(f"[EvalNet] Epoch {epoch}, Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loader:
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping")
                    break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model


# ===============================
# Main entry point
# ===============================
def main():
    parser = argparse.ArgumentParser(description="NNUE Pretrain + Finetune Pipeline")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--train_csv", default="/kaggle/input/train-your-own-stockfish-nnue/train.csv", help="CSV with FEN and Evaluation (for finetune)")
    parser.add_argument("--test_csv", default="/kaggle/input/train-your-own-stockfish-nnue/test.csv", help="CSV with FEN only")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--pretrained_path", help="Path to pretrained autoencoder .pt")
    parser.add_argument("--save_path", default="model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSVs
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    if "FEN" not in train_df.columns:
        raise RuntimeError("train.csv must contain 'FEN' column")

    has_eval = "Evaluation" in train_df.columns

    # Train/val split
    n = train_df.shape[0]
    n_val = int(n * args.val_split)
    train_split = train_df[:-n_val] if n_val > 0 else train_df
    val_split = train_df[-n_val:] if n_val > 0 else pd.DataFrame()

    # Encode
    X_train_np = encode_series_to_np(train_split["FEN"], "encode train").reshape(len(train_split), -1)
    X_train = torch.from_numpy(X_train_np).to(device)
    y_train = None
    if has_eval:
        y_train = torch.from_numpy(train_split["Evaluation"].values.astype(np.float32)).unsqueeze(1).to(device)

    X_val, y_val = None, None
    if n_val > 0:
        X_val_np = encode_series_to_np(val_split["FEN"], "encode val").reshape(len(val_split), -1)
        X_val = torch.from_numpy(X_val_np).to(device)
        if has_eval:
            y_val = torch.from_numpy(val_split["Evaluation"].values.astype(np.float32)).unsqueeze(1).to(device)

    # Mode: pretrain
    if args.mode == "pretrain":
        train_ds = TensorDataset(X_train, torch.zeros(len(X_train), 1))  # dummy labels
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        autoenc = ChessAutoencoder(input_dim=INPUT_DIM, latent_dim=args.latent_dim)
        autoenc = train_autoencoder(autoenc, train_loader, device, epochs=args.epochs)
        torch.save(autoenc.state_dict(), args.save_path)
        print(f"Saved pretrained autoencoder to {args.save_path}")

    # Mode: finetune
    elif args.mode == "finetune":
        if not args.pretrained_path:
            raise RuntimeError("--pretrained_path required for finetune")

        autoenc = ChessAutoencoder(input_dim=INPUT_DIM, latent_dim=args.latent_dim)
        autoenc.load_state_dict(torch.load(args.pretrained_path, map_location=device))
        model = EvalNet(autoenc, latent_dim=args.latent_dim)

        if y_train is None:
            raise RuntimeError("train.csv must have 'Evaluation' column for finetune")

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val) if n_val > 0 else None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

        model = train_evalnet(model, train_loader, val_loader, device, epochs=args.epochs)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved finetuned model to {args.save_path}")

        # --- Evaluate on val set ---
        if val_loader:
            model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb).cpu().numpy()[:, 0]
                    y_pred.append(preds)
                    y_true.append(yb.cpu().numpy()[:, 0])
            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
            mae_val = mean_absolute_error(y_true, y_pred)
            print("Validation MAE:", mae_val)

        # --- Predict test set ---
        if "FEN" in test_df.columns:
            X_test_np = encode_series_to_np(test_df["FEN"], "encode test").reshape(len(test_df), -1)
            X_test = torch.from_numpy(X_test_np).to(device)

            test_loader = DataLoader(TensorDataset(X_test), batch_size=args.batch_size, shuffle=False)
            preds_all = []
            model.eval()
            with torch.no_grad():
                for xb, in test_loader:
                    preds = model(xb).cpu().numpy()[:, 0]
                    preds_all.append(preds)
            preds_all = np.concatenate(preds_all)

            submission = pd.DataFrame({
                "FEN": test_df["FEN"],
                "Predicted_Evaluation": preds_all
            })
            submission.to_csv("submission.csv", index=False)
            print("Saved predictions to submission.csv")


if __name__ == "__main__":
    main()



!python nnue_pipeline.py --mode pretrain --train_csv /kaggle/input/train-your-own-stockfish-nnue/train.csv --epochs 20 --save_path autoenc.pt



!python nnue_pipeline.py --mode finetune --train_csv /kaggle/input/train-your-own-stockfish-nnue/train.csv --test_csv /kaggle/input/train-your-own-stockfish-nnue/test.csv --pretrained_path autoenc.pt --epochs 40 --save_path finetuned.pt



import torch
import pandas as pd
import numpy as np
from nnue_pipeline import encode_series_to_np, INPUT_DIM, ChessAutoencoder, EvalNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
test_csv = "/kaggle/input/train-your-own-stockfish-nnue/test.csv"
autoenc_path = "autoenc.pt"
finetuned_path = "finetuned.pt"
submission_path = "submission.csv"
latent_dim = 128

# Load test data
test_df = pd.read_csv(test_csv, sep="\t")
X_test_np = encode_series_to_np(test_df["FEN"], "encode test").reshape(len(test_df), -1)
X_test = torch.from_numpy(X_test_np).to(device)

# Load models
autoenc = ChessAutoencoder(input_dim=INPUT_DIM, latent_dim=latent_dim)
autoenc.load_state_dict(torch.load(autoenc_path, map_location=device))

model = EvalNet(autoenc, latent_dim=latent_dim)
model.load_state_dict(torch.load(finetuned_path, map_location=device))
model.to(device)
model.eval()

# Predict
from torch.utils.data import DataLoader, TensorDataset
test_loader = DataLoader(TensorDataset(X_test), batch_size=256)
preds_all = []
with torch.no_grad():
    for xb, in test_loader:
        preds = model(xb).cpu().numpy()[:, 0]
        preds_all.append(preds)
preds_all = np.concatenate(preds_all)

# Save submission
submission = pd.DataFrame({
    "FEN": test_df["FEN"],
    "Predicted_Evaluation": preds_all
})
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")



# from IPython.display import display
# from ipywidgets import FileUpload

# # Create upload widget
# upload_widget = FileUpload(accept='', multiple=False)  # accept='' means any file type
# display(upload_widget)



import torch
from nnue_pipeline import EvalNet, ChessAutoencoder, INPUT_DIM

device = torch.device("cpu")
latent_dim = 128

autoenc = ChessAutoencoder(input_dim=INPUT_DIM, latent_dim=latent_dim)
autoenc.load_state_dict(torch.load("autoenc.pt", map_location=device))

model = EvalNet(autoenc, latent_dim=latent_dim)
model.load_state_dict(torch.load("finetuned.pt", map_location=device))
model.eval()
print("pls")
# Dummy input
dummy_input = torch.zeros(1, INPUT_DIM)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "nnue_model.onnx",
    input_names=["board"],
    output_names=["evaluation"],
    dynamic_axes={"board": {0: "batch_size"}}
)



# # Save uploaded file
# def save_uploaded_file(upload_widget):
#     if upload_widget.value:
#         for uploaded_file in upload_widget.value:  # each is UploadedFile
#             filename = uploaded_file.name
#             content = uploaded_file.content
#             with open(filename, 'wb') as f:
#                 f.write(content)
#             print(f"Saved file: {filename}")
#     else:
#         print("No file uploaded yet.")

# # After uploading, run this to save
# save_uploaded_file(upload_widget)



%%writefile main.js
import * as onnx from 'onnxjs';
import { Chess } from 'chess.js';

const chess = new Chess();
const session = new onnx.InferenceSession();
await session.loadModel("./nnue_model.onnx");

function fenToInputVector(fen) {
    // Convert FEN to 64*13 one-hot flattened array
    // Must match your PyTorch preprocessing
    return new Float32Array(832);
}

async function evaluatePosition(fen) {
    const inputTensor = new onnx.Tensor(fenToInputVector(fen), 'float32', [1, 832]);
    const outputMap = await session.run({ board: inputTensor });
    return outputMap.evaluation.data[0]; // evaluation score
}

async function pickBestMove() {
    let bestMove = null;
    let bestEval = chess.turn() === 'w' ? -Infinity : Infinity;

    chess.moves().forEach(async (move) => {
        chess.move(move);
        const evalScore = await evaluatePosition(chess.fen());
        chess.undo();

        if ((chess.turn() === 'w' && evalScore > bestEval) ||
            (chess.turn() === 'b' && evalScore < bestEval)) {
            bestEval = evalScore;
            bestMove = move;
        }
    });

    chess.move(bestMove);
    return bestMove;
}





