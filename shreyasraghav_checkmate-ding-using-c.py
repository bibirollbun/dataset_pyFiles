%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments


# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)


%%writefile fast_chess.cpp
#include <Python.h>
#include <array>
#include <string>
#include <vector>
#include <cstring>

// Piece values for quick evaluation
const int PIECE_VALUES[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 100, 300, 300, 500, 900, 0, -100, -300, -300, -500, -900, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// Quick lookup for piece type
inline int get_piece_value(char piece) {
    switch(piece) {
        case 'P': return 100;
        case 'N': return 300;
        case 'B': return 300;
        case 'R': return 500;
        case 'Q': return 900;
        case 'K': return 10000;
        case 'p': return -100;
        case 'n': return -300;
        case 'b': return -300;
        case 'r': return -500;
        case 'q': return -900;
        case 'k': return -10000;
        default: return 0;
    }
}

// Fast move evaluation
inline int evaluate_move(const char* fen, const char* move) {
    int score = 0;
    
    // Piece positions for quick evaluation
    char board[64];
    int idx = 0;
    size_t fen_length = strlen(fen);
    
    for(size_t i = 0; i < fen_length && fen[i] != ' '; i++) {
        if(fen[i] >= '1' && fen[i] <= '8') {
            int empty = fen[i] - '0';
            while(empty-- && idx < 64) board[idx++] = ' ';
        }
        else if(fen[i] != '/') {
            if(idx < 64) board[idx++] = fen[i];
        }
    }
    
    // Convert move coordinates
    int to_file = move[2] - 'a';
    int to_rank = '8' - move[3];
    
    int to_idx = to_rank * 8 + to_file;
    
    // Bounds check
    if(to_idx >= 0 && to_idx < 64) {
        // Capture value
        char captured = board[to_idx];
        if(captured != ' ') {
            score += get_piece_value(captured);
        }
        
        // Center control bonus
        if((to_file == 3 || to_file == 4) && (to_rank == 3 || to_rank == 4)) {
            score += 10;
        }
        
        // Promotion bonus
        if(strlen(move) > 4 && move[4] == 'q') {
            score += 800;
        }
    }
    
    return score;
}

static PyObject* fast_evaluate(PyObject* self, PyObject* args) {
    const char* fen;
    const char* move;
    
    if (!PyArg_ParseTuple(args, "ss", &fen, &move)) {
        return NULL;
    }
    
    int score = evaluate_move(fen, move);
    return PyLong_FromLong(score);
}

static PyMethodDef FastChessMethods[] = {
    {"evaluate", fast_evaluate, METH_VARARGS, "Evaluate a chess move"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastchessmodule = {
    PyModuleDef_HEAD_INIT,
    "fastchess",
    NULL,
    -1,
    FastChessMethods
};

PyMODINIT_FUNC PyInit_fastchess(void) {
    return PyModule_Create(&fastchessmodule);
}


%%writefile setup.py
from setuptools import setup, Extension

module = Extension(
    'fastchess',
    sources=['fast_chess.cpp'],
    extra_compile_args=['-O3', '-march=native'],
    language='c++'
)

setup(
    name='fastchess',
    version='1.0',
    ext_modules=[module]
)


%%writefile main.py
from Chessnut import Game

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
    ' ': 0
}

# Piece position tables for evaluation
PIECE_POSITION_VALUES = {
    'P': [  # Pawn
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    'N': [  # Knight
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
}

def count_pieces(game):
    """Count major pieces on the board"""
    queens = 0
    major_pieces = 0
    for i in range(64):
        piece = game.board.get_piece(i)
        if piece.upper() == 'Q':
            queens += 1
        elif piece.upper() in 'RBN':
            major_pieces += 1
    return queens, major_pieces

def evaluate_position(game, move):
    """Evaluate a chess position considering multiple factors"""
    score = 0
    to_square = Game.xy2i(move[2:4])
    from_square = Game.xy2i(move[0:2])
    moving_piece = game.board.get_piece(from_square)
    captured_piece = game.board.get_piece(to_square)
    
    # Material value and captures
    if captured_piece != ' ':
        score += PIECE_VALUES[captured_piece]
        # Extra bonus for capturing with less valuable piece
        if abs(PIECE_VALUES[moving_piece]) < abs(PIECE_VALUES[captured_piece]):
            score += 50
    
    # Position evaluation using piece-square tables
    to_rank, to_file = to_square // 8, to_square % 8
    piece_type = moving_piece.upper()
    if piece_type in PIECE_POSITION_VALUES:
        position_value = PIECE_POSITION_VALUES[piece_type][to_rank * 8 + to_file]
        score += position_value if moving_piece.isupper() else -position_value

    # Development and center control in opening
    if piece_type in 'NB' and from_square in [1,2,5,6,57,58,61,62]:
        score += 30  # Piece development
    if piece_type == 'P' and to_square in [27,28,35,36]:  # Center control
        score += 20

    # King safety
    queens, pieces = count_pieces(game)
    is_endgame = queens == 0 or pieces <= 6
    if piece_type == 'K':
        if is_endgame:
            center_distance = abs(3.5 - to_file) + abs(3.5 - to_rank)
            score += (7 - center_distance) * 10  # King centralization in endgame
        else:
            if to_file in [0,1,6,7]:  # Keep king on the sides in middlegame
                score += 30

    # Pawn promotion
    if len(move) > 4 and move[4].lower() == 'q':
        score += 900

    return score

def chess_bot(obs):
    """
    Advanced chess bot with sophisticated evaluation
    """
    game = Game(obs.board)
    moves = list(game.get_moves())
    
    if not moves:
        return None

    # Check for immediate checkmate
    for move in moves[:5]:  # Limit check to first 5 moves for speed
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move

    # Find best move through evaluation
    best_move = moves[0]
    best_score = float('-inf')
    
    for move in moves:
        score = evaluate_position(game, move)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


!apt-get update && apt-get install -y python3-dev g++
!python3 -m pip install setuptools
!python3 -m pip install wheel


!python setup.py build_ext --inplace


result = env.run(["main.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=1000, height=1000)


%cd /kaggle/working
!tar -czvf submission.tar.gz main.py

