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


!pip install --upgrade kaggle-environments


!pip install --upgrade pip setuptools wheel



!pip install --upgrade pygame



!pip install python-chess


import importlib.resources as resources



!pip install --upgrade google-cloud google-auth sphinxcontrib-matlabdomain



!pip install --upgrade tensorflow



!pip install tensorflow==2.17.0


!pip install --upgrade jax tensorflow tensorflow-lite



!pip install --upgrade tensorflow-decision-forests tensorflow-text tf-keras



!pip uninstall -y tensorflow tensorflow-decision-forests tensorflow-text tf-keras
!pip install tensorflow==2.17.0 tensorflow-decision-forests tensorflow-text tf-keras



import chess
import chess.engine
import cv2
import pytesseract
import numpy as np

def get_fen_from_image(image_path):
    """Ekstrak FEN dari gambar menggunakan OCR dengan preprocessing."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing untuk meningkatkan akurasi OCR
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Mengurangi noise
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)  # Thresholding
    
    # Gunakan OCR dengan whitelist karakter yang relevan untuk FEN
    fen = pytesseract.image_to_string(img, config='--psm 6 -c tessedit_char_whitelist="rnbqkpRNBQKP12345678/ w-"')
    
    # Bersihkan hasil OCR
    fen = fen.strip().replace("\n", " ")

    # Validasi apakah FEN terbaca dengan benar
    if not fen or " " not in fen:
        print("OCR gagal membaca FEN dengan benar. Menggunakan posisi default.")
        return None
    
    return fen

def evaluate_board(board):
    """Fungsi sederhana untuk mengevaluasi posisi catur."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0  # Raja tidak diberi nilai karena tidak dikorbankan
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """Algoritma Minimax dengan Alpha-Beta Pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def get_best_move(board, depth=3):
    """Mendapatkan langkah terbaik menggunakan Minimax dengan Alpha-Beta Pruning."""
    best_move = None
    best_value = float('-inf')

    for move in board.legal_moves:
        board.push(move)
        move_value = minimax(board, depth - 1, float('-inf'), float('inf'), False)
        board.pop()

        if move_value > best_value:
            best_value = move_value
            best_move = move

    return best_move

def main(image_path):
    """Fungsi utama untuk membaca gambar, mengekstrak FEN, dan memainkan AI catur."""
    if image_path:
        fen = get_fen_from_image(image_path)
        if fen:
            try:
                board = chess.Board(fen)
            except ValueError:
                print(f"FEN tidak valid: {fen}. Menggunakan posisi default.")
                board = chess.Board()
        else:
            board = chess.Board()
    else:
        board = chess.Board()

    print("Posisi awal papan catur:")
    print(board)

    best_move = get_best_move(board)
    
    if best_move:
        print(f"Langkah terbaik yang dipilih: {best_move}")
        board.push(best_move)
    else:
        print("Tidak ada langkah yang tersedia!")

    print("Papan setelah langkah AI:")
    print(board)

if __name__ == "__main__":
    IMAGE_PATH = "/kaggle/input/fide-google-efficiency-chess-ai-challenge/Screenshot 2024-10-09 at 10.45.28AM.png"
    main(IMAGE_PATH)



import requests
requests.get('http://www.google.com', timeout=10).ok


from kaggle_environments import make
env = make("chess", debug=True)



result = env.run(["random", "random"])
env.render(mode="ipython", width=1000, height=1000)



%%writefile submission.py
from Chessnut import Game
import random

def chess_bot(obs):
    """
    Bot catur sederhana yang memprioritaskan checkmate, menangkap bidak, promosi ratu, lalu langkah acak.
    """
    game = Game(obs.board)
    moves = list(game.get_moves())

    # 1. Mencari langkah checkmate
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # 2. Menangkap bidak lawan
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move

    # 3. Melakukan promosi ke ratu jika memungkinkan
    for move in moves:
        if "q" in move.lower():
            return move

    # 4. Jika tidak ada langkah terbaik, pilih langkah secara acak
    return random.choice(moves)



result = env.run(["submission.py", "random"])
print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
env.render(mode="ipython", width=1000, height=1000)


