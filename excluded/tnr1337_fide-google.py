pip install python-chess kaggle-environments



%%writefile main.py
import chess
import random
import time
import math
import requests
from collections import defaultdict

# =============================================================================
# INTERNET BAĞLANTISI KONTROLÜ
# =============================================================================
try:
    if requests.get('http://www.google.com', timeout=10).ok:
        print("Internet connectivity check passed.")
except Exception as e:
    print("Internet connectivity check failed:", e)

# =============================================================================
# SABİTLER ve PARAMETRELER
# =============================================================================
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000
}

# Beyaz için piece-square tabloları (siyah için chess.square_mirror uygulanacak)
PAWN_TABLE = [
       0,   5,   5, -10, -10,   5,   5,   0,
       0,   5,   5,   0,   0,   5,   5,   0,
       0,   5,  10,  20,  20,  10,   5,   0,
       5,  10,  20,  30,  30,  20,  10,   5,
      10,  20,  30,  40,  40,  30,  20,  10,
      20,  30,  40,  50,  50,  40,  30,  20,
      30,  40,  50,  60,  60,  50,  40,  30,
       0,   0,   0,   0,   0,   0,   0,   0
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
       0,   0,   5,  10,  10,   5,   0,   0,
      -5,   0,   0,   0,   0,   0,   0,  -5,
      -5,   0,   0,   0,   0,   0,   0,  -5,
      -5,   0,   0,   0,   0,   0,   0,  -5,
      -5,   0,   0,   0,   0,   0,   0,  -5,
      -5,   0,   0,   0,   0,   0,   0,  -5,
       5,  10,  10,  10,  10,  10,  10,   5,
       0,   0,   0,   0,   0,   0,   0,   0,
]

QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE,
}

# =============================================================================
# TRANSPOZİSYON TABLOSU FLAG’LERİ ve Fonksiyonları
# =============================================================================
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2
transposition_table = {}

def tt_lookup(board):
    # Eğer mevcutsa Zobrist hash kullan, yoksa FEN
    key = board.zobrist_hash() if hasattr(board, "zobrist_hash") else board.fen()
    return transposition_table.get(key, None)

def tt_store(board, depth, value, flag, best_move):
    key = board.zobrist_hash() if hasattr(board, "zobrist_hash") else board.fen()
    transposition_table[key] = (depth, value, flag, best_move)

# =============================================================================
# KILLER & HISTORY HEURISTICS
# =============================================================================
killer_moves = defaultdict(lambda: [None, None])  # her ply için iki killer hamle
history_heuristic = defaultdict(int)

def order_moves(board, moves, ply):
    move_scores = []
    for move in moves:
        score = 0
        # MVV-LVA: Yakalama hamleleri için
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                attacker = board.piece_at(move.from_square)
                score += 1000 + PIECE_VALUES[captured.piece_type] - (PIECE_VALUES[attacker.piece_type] if attacker else 0)
        # Killer move bonus
        if move == killer_moves[ply][0]:
            score += 900
        elif move == killer_moves[ply][1]:
            score += 800
        # History heuristic
        score += history_heuristic[move]
        move_scores.append((score, move))
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [m for s, m in move_scores]

# =============================================================================
# DEĞERLENDİRME FONKSİYONU (Negamax uyumlu)
# =============================================================================
def evaluate_board(board):
    # Checkmate: hamle yapan tarafta mat varsa her zaman kötü
    if board.is_checkmate():
        return -float('inf')
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            table = PIECE_SQUARE_TABLES[piece.piece_type]
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            # Pozisyonel bonus: beyaz için artı, siyah için eksi
            score += (value + table[idx]) if piece.color == chess.WHITE else -(value + table[idx])
    # Mobilite: her iki tarafın hamle sayıları farkı
    current_turn = board.turn
    if current_turn == chess.WHITE:
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
    else:
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
    mobility_score = 10 * (white_mobility - black_mobility)
    score += mobility_score
    # Negamax dönüş: her zaman hamle yapan taraf açısından değerlendirme
    return score if board.turn == chess.WHITE else -score

# =============================================================================
# QUIESCENCE SEARCH (Negamax versiyonu)
# =============================================================================
def quiescence(board, alpha, beta, start_time, time_limit):
    if time.time() - start_time > time_limit:
        raise TimeoutError
    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            score = -quiescence(board, -beta, -alpha, start_time, time_limit)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
    return alpha

# =============================================================================
# NEGAMAX ALPHA-BETA ARAMA (Killer, History, TT, Aspiration Windows dahil)
# =============================================================================
def negamax(board, depth, alpha, beta, start_time, time_limit, ply):
    if time.time() - start_time > time_limit:
        raise TimeoutError

    tt_entry = tt_lookup(board)
    if tt_entry is not None:
        tt_depth, tt_value, tt_flag, tt_best_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_value
            elif tt_flag == LOWERBOUND:
                alpha = max(alpha, tt_value)
            elif tt_flag == UPPERBOUND:
                beta = min(beta, tt_value)
            if alpha >= beta:
                return tt_value

    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, start_time, time_limit)

    original_alpha = alpha

    # Kök düğümde (ply==0) Aspiration Window (PV search) uygulanıyor
    if ply == 0:
        window = 50
        a, b = alpha, beta
        score = negamax(board, depth, a, b, start_time, time_limit, ply + 1)
        while score <= a or score >= b:
            if score <= a:
                a -= window
            if score >= b:
                b += window
            score = negamax(board, depth, a, b, start_time, time_limit, ply + 1)
        return score

    best_value = -float('inf')
    best_move = None
    moves = list(board.legal_moves)
    moves_ordered = order_moves(board, moves, ply)

    for move in moves_ordered:
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, start_time, time_limit, ply + 1)
        except TimeoutError:
            board.pop()
            raise TimeoutError
        board.pop()
        if score > best_value:
            best_value = score
            best_move = move
        alpha = max(alpha, score)
        history_heuristic[move] += depth * depth
        if alpha >= beta:
            # Beta cutoff: Killer hamle güncellemesi
            if move not in killer_moves[ply]:
                killer_moves[ply][1] = killer_moves[ply][0]
                killer_moves[ply][0] = move
            break

    flag = EXACT
    if best_value <= original_alpha:
        flag = UPPERBOUND
    elif best_value >= beta:
        flag = LOWERBOUND
    tt_store(board, depth, best_value, flag, best_move)
    return best_value

# =============================================================================
# MTD(f) ALGORITMASI (Negamax çağırıyor)
# =============================================================================
def mtd_f(board, first_guess, depth, time_limit, start_time):
    g = first_guess
    upperBound = float('inf')
    lowerBound = -float('inf')
    while lowerBound < upperBound:
        beta = g if g == lowerBound else g + 1
        g = negamax(board, depth, beta - 1, beta, start_time, time_limit, ply=0)
        if g < beta:
            upperBound = g
        else:
            lowerBound = g
    return g

# =============================================================================
# ITERATİF DERİNLEME ARAMA (MTD(f) ile entegre)
# =============================================================================
def iterative_deepening(board, max_time):
    start_time = time.time()
    best_move = None
    depth = 1
    first_guess = 0
    while True:
        try:
            score = mtd_f(board, first_guess, depth, max_time, start_time)
            first_guess = score
            # Transpozisyon tablosundaki en iyi hamleyi kullan
            tt_entry = tt_lookup(board)
            if tt_entry:
                best_move = tt_entry[3]
            depth += 1
        except TimeoutError:
            break
    return best_move

# =============================================================================
# ANA BOT FONKSİYONU (Kaggle Submission İçin)
# =============================================================================
def chess_bot(obs):
    board = chess.Board(obs.board)
    # Zaman yönetimi: kalan sürenin %10'u, minimum 0.5 saniye kullanılsın
    time_limit = max(0.5, obs.remainingOverageTime * 0.1)
    try:
        best_move = iterative_deepening(board, time_limit)
        if best_move is None:
            best_move = random.choice(list(board.legal_moves))
    except Exception:
        best_move = random.choice(list(board.legal_moves))
    return best_move.uci()

# =============================================================================
# SIMÜLASYON & DETAYLI LOG ÇIKTILARI (Yerel Test için)
# =============================================================================
if __name__ == '__main__':
    # Basit test: başlangıç FEN'i ile engine çalıştırılır.
    test_board = chess.Board()
    class DummyObs:
        board = test_board.fen()
        mark = "white"
        remainingOverageTime = 10
    print("Simülasyon Testi - Seçilen hamle:", chess_bot(DummyObs()))
    
    # Kaggle Environments ile simülasyon oyunu
    try:
        from kaggle_environments import make
        print("\nKaggle Environment Simülasyonu Başlatılıyor...")
        env = make("chess", debug=True)
        result = env.run(["submission.py", "random"])
        print("\nSimülasyon Oyunu Sonuçları:")
        for i, agent in enumerate(result[-1]):
            print(f"\tAgent {i}: {agent.status} / {agent.reward} / {agent.observation.remainingOverageTime}")
        env.render(mode="ipython", width=1000, height=1000)
    except Exception as e:
        print("Kaggle environment simülasyonu yapılamadı:", e)



!tar -czf submission.tar.gz main.py


