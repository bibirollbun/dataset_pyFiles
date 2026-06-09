import os
import glob
from copy import deepcopy
from code_golf_utils import *

PUZZLES_PATH = "dataset\google-code-golf-2025"
SOLUTIONS_PATH = "tasks"
LABEL = "train"
solutions = glob.glob(os.path.join(SOLUTIONS_PATH, '*.py'))
print(f"Number of puzzle files: {len(glob.glob(os.path.join(PUZZLES_PATH, '*.json')))}")
print(f"Number of solutions: {len(solutions)}")
print(f"Uncompressed Competition score: {sum([max(1, 2500-os.path.getsize(f)) for f in solutions])}")
print(f"Uncompressed Average filesize: {sum([os.path.getsize(f) for f in solutions])/len(solutions) :.4f} bytes")


def plot(X, ax):
    X=[list(tuple(row)) for row in X]
    for r in range(len(X)):
        for c in range(len(X[0])):
            X[r][c] = colors[X[r][c]]
    ax.imshow(X)


puzzle_data = load_examples(1)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(1, puzzle_data)


puzzle_data = load_examples(2)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(2, puzzle_data)


puzzle_data = load_examples(3)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(3, puzzle_data)


puzzle_data = load_examples(4)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(4, puzzle_data)


puzzle_data = load_examples(5)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(5, puzzle_data)


puzzle_data = load_examples(6)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(6, puzzle_data)


puzzle_data = load_examples(7)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(7, puzzle_data)


puzzle_data = load_examples(8)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(8, puzzle_data)


puzzle_data = load_examples(9)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(9, puzzle_data)


puzzle_data = load_examples(10)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(10, puzzle_data)


puzzle_data = load_examples(11)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(11, puzzle_data)


puzzle_data = load_examples(12)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(12, puzzle_data)


puzzle_data = load_examples(13)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(13, puzzle_data)


puzzle_data = load_examples(14)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(14, puzzle_data)


puzzle_data = load_examples(15)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(15, puzzle_data)


puzzle_data = load_examples(16)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(16, puzzle_data)


puzzle_data = load_examples(17)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:5])

verify_program(17, puzzle_data)


puzzle_data = load_examples(18)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(18, puzzle_data)


puzzle_data = load_examples(19)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(19, puzzle_data)


puzzle_data = load_examples(20)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(20, puzzle_data)


puzzle_data = load_examples(21)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(21, puzzle_data)


puzzle_data = load_examples(22)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(22, puzzle_data)


puzzle_data = load_examples(23)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(23, puzzle_data)


puzzle_data = load_examples(24)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(24, puzzle_data)


puzzle_data = load_examples(25)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(25, puzzle_data)


puzzle_data = load_examples(26)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(26, puzzle_data)


puzzle_data = load_examples(27)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[19:21])

verify_program(27, puzzle_data)


puzzle_data = load_examples(28)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(28, puzzle_data)


puzzle_data = load_examples(29)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[7:10])

verify_program(29, puzzle_data)


puzzle_data = load_examples(30)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(30, puzzle_data)


puzzle_data = load_examples(31)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(31, puzzle_data)


puzzle_data = load_examples(32)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(32, puzzle_data)


puzzle_data = load_examples(33)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(33, puzzle_data)


puzzle_data = load_examples(34)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(34, puzzle_data)


puzzle_data = load_examples(35)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(35, puzzle_data)


puzzle_data = load_examples(36)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(36, puzzle_data)


puzzle_data = load_examples(37)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(37, puzzle_data)


puzzle_data = load_examples(38)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(38, puzzle_data)


puzzle_data = load_examples(39)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(39, puzzle_data)


puzzle_data = load_examples(40)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(40, puzzle_data)


puzzle_data = load_examples(41)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(41, puzzle_data)


puzzle_data = load_examples(42)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(42, puzzle_data)


puzzle_data = load_examples(43)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(43, puzzle_data)


puzzle_data = load_examples(44)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(44, puzzle_data)


puzzle_data = load_examples(45)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(45, puzzle_data)


puzzle_data = load_examples(46)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(46, puzzle_data)


puzzle_data = load_examples(47)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(47, puzzle_data)


puzzle_data = load_examples(48)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(48, puzzle_data)


puzzle_data = load_examples(49)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(49, puzzle_data)


puzzle_data = load_examples(50)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(50, puzzle_data)


puzzle_data = load_examples(51)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(51, puzzle_data)


puzzle_data = load_examples(52)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(52, puzzle_data)


puzzle_data = load_examples(53)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(53, puzzle_data)


puzzle_data = load_examples(54)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:2])

verify_program(54, puzzle_data)


puzzle_data = load_examples(55)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(55, puzzle_data)


puzzle_data = load_examples(56)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(56, puzzle_data)


puzzle_data = load_examples(57)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(57, puzzle_data)


puzzle_data = load_examples(58)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[0:5])

verify_program(58, puzzle_data)


puzzle_data = load_examples(59)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(59, puzzle_data)


puzzle_data = load_examples(60)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(60, puzzle_data)


puzzle_data = load_examples(61)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(61, puzzle_data)


puzzle_data = load_examples(62)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(62, puzzle_data)


puzzle_data = load_examples(63)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(63, puzzle_data)


puzzle_data = load_examples(64)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(64, puzzle_data)


puzzle_data = load_examples(65)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(65, puzzle_data)


puzzle_data = load_examples(66)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(66, puzzle_data)


puzzle_data = load_examples(67)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(67, puzzle_data)


puzzle_data = load_examples(68)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(68, puzzle_data)


puzzle_data = load_examples(69)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(69, puzzle_data)


puzzle_data = load_examples(70)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(70, puzzle_data)


puzzle_data = load_examples(71)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(71, puzzle_data)


puzzle_data = load_examples(72)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(72, puzzle_data)


puzzle_data = load_examples(73)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(73, puzzle_data)


puzzle_data = load_examples(74)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(74, puzzle_data)


puzzle_data = load_examples(75)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(75, puzzle_data)


puzzle_data = load_examples(76)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(76, puzzle_data)


puzzle_data = load_examples(77)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(77, puzzle_data)


puzzle_data = load_examples(78)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(78, puzzle_data)


puzzle_data = load_examples(79)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(79, puzzle_data)


puzzle_data = load_examples(80)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(80, puzzle_data)


puzzle_data = load_examples(81)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(81, puzzle_data)


puzzle_data = load_examples(82)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(82, puzzle_data)


puzzle_data = load_examples(83)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(83, puzzle_data)


puzzle_data = load_examples(84)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(84, puzzle_data)


puzzle_data = load_examples(85)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(85, puzzle_data)


puzzle_data = load_examples(86)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(86, puzzle_data)


puzzle_data = load_examples(87)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(87, puzzle_data)


puzzle_data = load_examples(88)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(88, puzzle_data)


puzzle_data = load_examples(89)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[21:24])

verify_program(89, puzzle_data)


puzzle_data = load_examples(90)
label = "test"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(90, puzzle_data)


puzzle_data = load_examples(91)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(91, puzzle_data)


puzzle_data = load_examples(92)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(92, puzzle_data)


puzzle_data = load_examples(93)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(93, puzzle_data)


puzzle_data = load_examples(94)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[259:])

verify_program(94, puzzle_data)


puzzle_data = load_examples(95)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(95, puzzle_data)


puzzle_data = load_examples(96)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(96, puzzle_data)


puzzle_data = load_examples(97)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(97, puzzle_data)


puzzle_data = load_examples(98)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(98, puzzle_data)


puzzle_data = load_examples(99)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(99, puzzle_data)


puzzle_data = load_examples(100)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(100, puzzle_data)


puzzle_data = load_examples(101)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(101, puzzle_data)


puzzle_data = load_examples(102)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(102, puzzle_data)


puzzle_data = load_examples(103)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(103, puzzle_data)


puzzle_data = load_examples(104)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(104, puzzle_data)


puzzle_data = load_examples(105)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[80:83])

verify_program(105, puzzle_data)


puzzle_data = load_examples(106)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(106, puzzle_data)


puzzle_data = load_examples(107)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(107, puzzle_data)


puzzle_data = load_examples(108)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(108, puzzle_data)


puzzle_data = load_examples(109)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(109, puzzle_data)


puzzle_data = load_examples(110)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(110, puzzle_data)


puzzle_data = load_examples(111)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(111, puzzle_data)


puzzle_data = load_examples(112)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(112, puzzle_data)


puzzle_data = load_examples(113)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(113, puzzle_data)


puzzle_data = load_examples(114)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(114, puzzle_data)


puzzle_data = load_examples(115)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(115, puzzle_data)


puzzle_data = load_examples(116)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(116, puzzle_data)


puzzle_data = load_examples(117)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(117, puzzle_data)


puzzle_data = load_examples(118)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(118, puzzle_data)


puzzle_data = load_examples(119)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(119, puzzle_data)


puzzle_data = load_examples(120)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(120, puzzle_data)


puzzle_data = load_examples(121)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:4])

verify_program(121, puzzle_data)


puzzle_data = load_examples(122)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(122, puzzle_data)


puzzle_data = load_examples(123)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(123, puzzle_data)


puzzle_data = load_examples(124)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(124, puzzle_data)


puzzle_data = load_examples(125)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(125, puzzle_data)


puzzle_data = load_examples(126)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(126, puzzle_data)


puzzle_data = load_examples(127)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(127, puzzle_data)


puzzle_data = load_examples(128)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(128, puzzle_data)


puzzle_data = load_examples(129)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(129, puzzle_data)


puzzle_data = load_examples(130)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(130, puzzle_data)


puzzle_data = load_examples(131)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(131, puzzle_data)


puzzle_data = load_examples(132)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(132, puzzle_data)


puzzle_data = load_examples(133)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(133, puzzle_data)


puzzle_data = load_examples(134)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(134, puzzle_data)


puzzle_data = load_examples(135)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(135, puzzle_data)


puzzle_data = load_examples(136)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(136, puzzle_data)


puzzle_data = load_examples(137)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(137, puzzle_data)


puzzle_data = load_examples(138)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(138, puzzle_data)


puzzle_data = load_examples(139)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(139, puzzle_data)


puzzle_data = load_examples(140)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(140, puzzle_data)


puzzle_data = load_examples(141)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(141, puzzle_data)


puzzle_data = load_examples(142)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(142, puzzle_data)


puzzle_data = load_examples(143)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(143, puzzle_data)


puzzle_data = load_examples(144)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(144, puzzle_data)


puzzle_data = load_examples(145)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(145, puzzle_data)


puzzle_data = load_examples(146)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(146, puzzle_data)


puzzle_data = load_examples(147)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(147, puzzle_data)


puzzle_data = load_examples(148)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(148, puzzle_data)


puzzle_data = load_examples(149)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(149, puzzle_data)


puzzle_data = load_examples(150)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(150, puzzle_data)


puzzle_data = load_examples(151)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(151, puzzle_data)


puzzle_data = load_examples(152)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(152, puzzle_data)


puzzle_data = load_examples(153)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(153, puzzle_data)


puzzle_data = load_examples(154)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(154, puzzle_data)


puzzle_data = load_examples(155)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(155, puzzle_data)


puzzle_data = load_examples(156)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(156, puzzle_data)


puzzle_data = load_examples(157)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(157, puzzle_data)


puzzle_data = load_examples(158)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(158, puzzle_data)


puzzle_data = load_examples(159)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(159, puzzle_data)


puzzle_data = load_examples(160)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(160, puzzle_data)


puzzle_data = load_examples(161)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(161, puzzle_data)


puzzle_data = load_examples(162)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(162, puzzle_data)


puzzle_data = load_examples(163)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(163, puzzle_data)


puzzle_data = load_examples(164)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(164, puzzle_data)


puzzle_data = load_examples(165)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(165, puzzle_data)


puzzle_data = load_examples(166)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(166, puzzle_data)


puzzle_data = load_examples(167)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(167, puzzle_data)


puzzle_data = load_examples(168)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(168, puzzle_data)


puzzle_data = load_examples(169)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(169, puzzle_data)


puzzle_data = load_examples(170)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(170, puzzle_data)


puzzle_data = load_examples(171)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(171, puzzle_data)


puzzle_data = load_examples(172)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(172, puzzle_data)


puzzle_data = load_examples(173)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(173, puzzle_data)


puzzle_data = load_examples(174)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(174, puzzle_data)


puzzle_data = load_examples(175)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(175, puzzle_data)


puzzle_data = load_examples(176)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(176, puzzle_data)


puzzle_data = load_examples(177)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(177, puzzle_data)


puzzle_data = load_examples(178)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(178, puzzle_data)


puzzle_data = load_examples(179)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(179, puzzle_data)


puzzle_data = load_examples(180)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(180, puzzle_data)


puzzle_data = load_examples(181)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(181, puzzle_data)


puzzle_data = load_examples(182)
label = "test"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(182, puzzle_data)


puzzle_data = load_examples(183)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(183, puzzle_data)


puzzle_data = load_examples(184)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(184, puzzle_data)


puzzle_data = load_examples(185)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(185, puzzle_data)


puzzle_data = load_examples(186)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(186, puzzle_data)


puzzle_data = load_examples(187)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(187, puzzle_data)


puzzle_data = load_examples(188)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(188, puzzle_data)


puzzle_data = load_examples(189)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(189, puzzle_data)


puzzle_data = load_examples(190)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(190, puzzle_data)


puzzle_data = load_examples(191)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[2:4])

# TODO
verify_program(191, puzzle_data)


puzzle_data = load_examples(192)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(192, puzzle_data)


puzzle_data = load_examples(193)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(193, puzzle_data)


puzzle_data = load_examples(194)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(194, puzzle_data)


puzzle_data = load_examples(195)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(195, puzzle_data)


puzzle_data = load_examples(196)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(196, puzzle_data)


puzzle_data = load_examples(197)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(197, puzzle_data)


puzzle_data = load_examples(198)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(198, puzzle_data)


puzzle_data = load_examples(199)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(199, puzzle_data)


puzzle_data = load_examples(200)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(200, puzzle_data)


puzzle_data = load_examples(201)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(201, puzzle_data)


puzzle_data = load_examples(202)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(202, puzzle_data)


puzzle_data = load_examples(203)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(203, puzzle_data)


puzzle_data = load_examples(204)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(204, puzzle_data)


puzzle_data = load_examples(205)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# Already solved but takes time (8min,51.6secs) (252 bytes, needs optimization)
verify_program(205, puzzle_data)


puzzle_data = load_examples(206)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(206, puzzle_data)


puzzle_data = load_examples(207)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(207, puzzle_data)


puzzle_data = load_examples(208)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(208, puzzle_data)


puzzle_data = load_examples(209)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(209, puzzle_data)


puzzle_data = load_examples(210)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(210, puzzle_data)


puzzle_data = load_examples(211)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(211, puzzle_data)


puzzle_data = load_examples(212)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(212, puzzle_data)


puzzle_data = load_examples(213)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(213, puzzle_data)


puzzle_data = load_examples(214)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(214, puzzle_data)


puzzle_data = load_examples(215)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(215, puzzle_data)


puzzle_data = load_examples(216)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(216, puzzle_data)


puzzle_data = load_examples(217)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(217, puzzle_data)


puzzle_data = load_examples(218)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(218, puzzle_data)


puzzle_data = load_examples(219)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(219, puzzle_data)


puzzle_data = load_examples(220)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(220, puzzle_data)


puzzle_data = load_examples(221)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(221, puzzle_data)


puzzle_data = load_examples(222)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(222, puzzle_data)


puzzle_data = load_examples(223)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(223, puzzle_data)


puzzle_data = load_examples(224)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(224, puzzle_data)


puzzle_data = load_examples(225)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(225, puzzle_data)


puzzle_data = load_examples(226)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(226, puzzle_data)


puzzle_data = load_examples(227)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(227, puzzle_data)


puzzle_data = load_examples(228)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(228, puzzle_data)


puzzle_data = load_examples(229)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(229, puzzle_data)


puzzle_data = load_examples(230)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(230, puzzle_data)


puzzle_data = load_examples(231)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(231, puzzle_data)


puzzle_data = load_examples(232)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(232, puzzle_data)


puzzle_data = load_examples(233)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(233, puzzle_data)


puzzle_data = load_examples(234)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(234, puzzle_data)


puzzle_data = load_examples(235)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(235, puzzle_data)


puzzle_data = load_examples(236)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(236, puzzle_data)


puzzle_data = load_examples(237)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(237, puzzle_data)


puzzle_data = load_examples(238)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(238, puzzle_data)


puzzle_data = load_examples(239)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(239, puzzle_data)


puzzle_data = load_examples(240)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(240, puzzle_data)


puzzle_data = load_examples(241)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(241, puzzle_data)


puzzle_data = load_examples(242)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(242, puzzle_data)


puzzle_data = load_examples(243)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(243, puzzle_data)


puzzle_data = load_examples(244)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(244, puzzle_data)


puzzle_data = load_examples(245)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(245, puzzle_data)


puzzle_data = load_examples(246)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(246, puzzle_data)


puzzle_data = load_examples(247)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(247, puzzle_data)


puzzle_data = load_examples(248)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(248, puzzle_data)


puzzle_data = load_examples(249)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(249, puzzle_data)


puzzle_data = load_examples(250)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(250, puzzle_data)


puzzle_data = load_examples(251)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(251, puzzle_data)


puzzle_data = load_examples(252)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(252, puzzle_data)


puzzle_data = load_examples(253)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(253, puzzle_data)


puzzle_data = load_examples(254)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(254, puzzle_data)


puzzle_data = load_examples(255)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(255, puzzle_data)


puzzle_data = load_examples(256)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(256, puzzle_data)


puzzle_data = load_examples(257)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(257, puzzle_data)


puzzle_data = load_examples(258)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(258, puzzle_data)


puzzle_data = load_examples(259)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(259, puzzle_data)


puzzle_data = load_examples(260)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(260, puzzle_data)


puzzle_data = load_examples(261)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(261, puzzle_data)


puzzle_data = load_examples(262)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(262, puzzle_data)


puzzle_data = load_examples(263)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(263, puzzle_data)


puzzle_data = load_examples(264)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(264, puzzle_data)


puzzle_data = load_examples(265)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(265, puzzle_data)


puzzle_data = load_examples(266)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(266, puzzle_data)


puzzle_data = load_examples(267)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(267, puzzle_data)


puzzle_data = load_examples(268)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(268, puzzle_data)


puzzle_data = load_examples(269)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(269, puzzle_data)


puzzle_data = load_examples(270)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(270, puzzle_data)


puzzle_data = load_examples(271)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(271, puzzle_data)


puzzle_data = load_examples(272)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(272, puzzle_data)


puzzle_data = load_examples(273)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(273, puzzle_data)


puzzle_data = load_examples(274)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(274, puzzle_data)


puzzle_data = load_examples(275)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(275, puzzle_data)


puzzle_data = load_examples(276)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(276, puzzle_data)


puzzle_data = load_examples(277)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(277, puzzle_data)


puzzle_data = load_examples(278)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# NEEDS TO BE REDONE
verify_program(278, puzzle_data)


puzzle_data = load_examples(279)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(279, puzzle_data)


puzzle_data = load_examples(280)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(280, puzzle_data)


puzzle_data = load_examples(281)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(281, puzzle_data)


puzzle_data = load_examples(282)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(282, puzzle_data)


puzzle_data = load_examples(283)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(283, puzzle_data)


puzzle_data = load_examples(284)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(284, puzzle_data)


puzzle_data = load_examples(285)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(285, puzzle_data)


puzzle_data = load_examples(286)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(286, puzzle_data)


puzzle_data = load_examples(287)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(287, puzzle_data)


puzzle_data = load_examples(288)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(288, puzzle_data)


puzzle_data = load_examples(289)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(289, puzzle_data)


puzzle_data = load_examples(290)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(290, puzzle_data)


puzzle_data = load_examples(291)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(291, puzzle_data)


puzzle_data = load_examples(292)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(292, puzzle_data)


puzzle_data = load_examples(293)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(293, puzzle_data)


puzzle_data = load_examples(294)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(294, puzzle_data)


puzzle_data = load_examples(295)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(295, puzzle_data)


puzzle_data = load_examples(296)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(296, puzzle_data)


puzzle_data = load_examples(297)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(297, puzzle_data)


puzzle_data = load_examples(298)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(298, puzzle_data)


puzzle_data = load_examples(299)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(299, puzzle_data)


puzzle_data = load_examples(300)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(300, puzzle_data)


puzzle_data = load_examples(301)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(301, puzzle_data)


puzzle_data = load_examples(302)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(302, puzzle_data)


puzzle_data = load_examples(303)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(303, puzzle_data)


puzzle_data = load_examples(304)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(304, puzzle_data)


puzzle_data = load_examples(305)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(305, puzzle_data)


puzzle_data = load_examples(306)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(306, puzzle_data)


puzzle_data = load_examples(307)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(307, puzzle_data)


puzzle_data = load_examples(308)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(308, puzzle_data)


puzzle_data = load_examples(309)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(309, puzzle_data)


puzzle_data = load_examples(310)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(310, puzzle_data)


puzzle_data = load_examples(311)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(311, puzzle_data)


puzzle_data = load_examples(312)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(312, puzzle_data)


puzzle_data = load_examples(313)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(313, puzzle_data)


puzzle_data = load_examples(314)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(314, puzzle_data)


puzzle_data = load_examples(315)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(315, puzzle_data)


puzzle_data = load_examples(316)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(316, puzzle_data)


puzzle_data = load_examples(317)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(317, puzzle_data)


puzzle_data = load_examples(318)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(318, puzzle_data)


puzzle_data = load_examples(319)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(319, puzzle_data)


puzzle_data = load_examples(320)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(320, puzzle_data)


puzzle_data = load_examples(321)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(321, puzzle_data)


puzzle_data = load_examples(322)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(322, puzzle_data)


puzzle_data = load_examples(323)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(323, puzzle_data)


puzzle_data = load_examples(324)
label = "arc-gen"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(324, puzzle_data)


puzzle_data = load_examples(325)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(325, puzzle_data)


puzzle_data = load_examples(326)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(326, puzzle_data)


puzzle_data = load_examples(327)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(327, puzzle_data)


puzzle_data = load_examples(328)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(328, puzzle_data)


puzzle_data = load_examples(329)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(329, puzzle_data)


puzzle_data = load_examples(330)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(330, puzzle_data)


puzzle_data = load_examples(331)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(331, puzzle_data)


puzzle_data = load_examples(332)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(332, puzzle_data)


puzzle_data = load_examples(333)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(333, puzzle_data)


puzzle_data = load_examples(334)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(334, puzzle_data)


puzzle_data = load_examples(335)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(335, puzzle_data)


puzzle_data = load_examples(336)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(336, puzzle_data)


puzzle_data = load_examples(337)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(337, puzzle_data)


puzzle_data = load_examples(338)
label = "test"
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(338, puzzle_data)


puzzle_data = load_examples(339)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(339, puzzle_data)


puzzle_data = load_examples(340)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(340, puzzle_data)


puzzle_data = load_examples(341)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(341, puzzle_data)


puzzle_data = load_examples(342)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(342, puzzle_data)


puzzle_data = load_examples(343)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

verify_program(343, puzzle_data)


puzzle_data = load_examples(344)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(344, puzzle_data)


puzzle_data = load_examples(345)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(345, puzzle_data)


puzzle_data = load_examples(346)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(346, puzzle_data)


puzzle_data = load_examples(347)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(347, puzzle_data)


puzzle_data = load_examples(348)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(348, puzzle_data)


puzzle_data = load_examples(349)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(349, puzzle_data)


puzzle_data = load_examples(350)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(350, puzzle_data)


puzzle_data = load_examples(351)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(351, puzzle_data)


puzzle_data = load_examples(352)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(352, puzzle_data)


puzzle_data = load_examples(353)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(353, puzzle_data)


puzzle_data = load_examples(354)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(354, puzzle_data)


puzzle_data = load_examples(355)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(355, puzzle_data)


puzzle_data = load_examples(356)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(356, puzzle_data)


puzzle_data = load_examples(357)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

# TODO
verify_program(357, puzzle_data)


puzzle_data = load_examples(358)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(358, puzzle_data)


puzzle_data = load_examples(359)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(359, puzzle_data)


puzzle_data = load_examples(360)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(360, puzzle_data)


puzzle_data = load_examples(361)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(361, puzzle_data)


puzzle_data = load_examples(362)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(362, puzzle_data)


puzzle_data = load_examples(363)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(363, puzzle_data)


puzzle_data = load_examples(364)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(364, puzzle_data)


puzzle_data = load_examples(365)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(365, puzzle_data)


puzzle_data = load_examples(366)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(366, puzzle_data)


puzzle_data = load_examples(367)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(367, puzzle_data)


puzzle_data = load_examples(368)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(368, puzzle_data)


puzzle_data = load_examples(369)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(369, puzzle_data)


puzzle_data = load_examples(370)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(370, puzzle_data)


puzzle_data = load_examples(371)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(371, puzzle_data)


puzzle_data = load_examples(372)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(372, puzzle_data)


puzzle_data = load_examples(373)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(373, puzzle_data)


puzzle_data = load_examples(374)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(374, puzzle_data)


puzzle_data = load_examples(375)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(375, puzzle_data)


puzzle_data = load_examples(376)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(376, puzzle_data)


puzzle_data = load_examples(377)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(377, puzzle_data)


puzzle_data = load_examples(378)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(378, puzzle_data)


puzzle_data = load_examples(379)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(379, puzzle_data)


puzzle_data = load_examples(380)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(380, puzzle_data)


puzzle_data = load_examples(381)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(381, puzzle_data)


puzzle_data = load_examples(382)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(382, puzzle_data)


puzzle_data = load_examples(383)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles[:3])

# TODO
verify_program(383, puzzle_data)


puzzle_data = load_examples(384)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(384, puzzle_data)


puzzle_data = load_examples(385)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(385, puzzle_data)


puzzle_data = load_examples(386)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(386, puzzle_data)


puzzle_data = load_examples(387)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(387, puzzle_data)


puzzle_data = load_examples(388)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(388, puzzle_data)


puzzle_data = load_examples(389)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(389, puzzle_data)


puzzle_data = load_examples(390)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(390, puzzle_data)


puzzle_data = load_examples(391)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(391, puzzle_data)


puzzle_data = load_examples(392)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(392, puzzle_data)


puzzle_data = load_examples(393)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(393, puzzle_data)


puzzle_data = load_examples(394)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(394, puzzle_data)


puzzle_data = load_examples(395)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(395, puzzle_data)


puzzle_data = load_examples(396)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(396, puzzle_data)


puzzle_data = load_examples(397)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(397, puzzle_data)


puzzle_data = load_examples(398)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(398, puzzle_data)


puzzle_data = load_examples(399)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(399, puzzle_data)


puzzle_data = load_examples(400)
label = LABEL
puzzles = puzzle_data[label]
print(f"num problems: {len(puzzles)}")
show_examples(puzzles)

verify_program(400, puzzle_data)

