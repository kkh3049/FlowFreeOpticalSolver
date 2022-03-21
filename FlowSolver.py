import mzucker_flowSolver as mzf

colorChars = list(mzf.ANSI_LOOKUP.keys())


def ConvertToMZuckForm(board):
    mzBoard = ''
    for y in range(len(board[0])):
        for x in range(len(board)):
            val, isCircle = board[x][y]
            charInBoard = '.' if val < 0 else colorChars[val]
            mzBoard += charInBoard
        mzBoard += '\n'
    return mzBoard


def ConvertFromMZuckDecoded(mzDecoded, originalBoard):
    mzColorToOurColor = {}
    ourBoard = []
    for x in range(len(mzDecoded[0])):
        ourCol = []
        for y in range(len(mzDecoded)):
            mzColor, mzJunction = mzDecoded[y][x]
            isCircle = mzJunction < 0
            ourCol.append((mzColor, isCircle))
            if isCircle:
                mzColorToOurColor[mzColor] = originalBoard[x][y][0]
        ourBoard.append(ourCol)
    ourBoard = [[(mzColorToOurColor[mzColor], isCircle) for mzColor, isCircle in ourCol] for ourCol in ourBoard]
    return ourBoard


def TracePath(board, pos):
    color, isCircle = board[pos[0]][pos[1]]
    assert isCircle
    visited = set()
    path = [pos]
    visited.add(pos)
    curPos = pos
    while True:
        nextPositions = [option for option in GetNeighbors(board, curPos) if board[option[0]][option[1]][0] == color and option not in visited]
        assert len(nextPositions) == 1
        curPos = nextPositions[0]
        visited.add(curPos)
        path.append(curPos)
        if board[curPos[0]][curPos[1]][1] == isCircle:
            break
    return path

def DetectPaths(ourBoard, numColors):
    colorSet = set()
    paths = [[] for _ in range(numColors)]
    for x, col in enumerate(ourBoard):
        for y, (cellColor, cellIsCircle) in enumerate(col):
            if cellIsCircle and cellColor not in colorSet:
                colorSet.add(cellColor)
                paths[cellColor] = TracePath(ourBoard, (x, y))
    return paths


def Solve(board):
    mzBoard = ConvertToMZuckForm(board)
    options = type('',(object,),{"quiet": False, "display_cycles": False, "display_color": False})()
    puzzle, colors = mzf.parse_puzzle(options, mzBoard)
    color_var, dir_vars, num_vars, clauses, reduce_time = \
        mzf.reduce_to_sat(options, puzzle, colors)
    sol, decoded, repairs, solve_time = mzf.solve_sat(options, puzzle, colors,
                                                      color_var, dir_vars, clauses)
    total_time = reduce_time + solve_time
    ourBoard = ConvertFromMZuckDecoded(decoded, board)
    paths = DetectPaths(ourBoard, len(colors))
    return ourBoard, paths


def SolveSlow(board):
    numColors = -1
    for y, row in enumerate(board):
        for x, square in enumerate(row):
            if square[0] > numColors:
                numColors = square[0]
    numColors += 1
    sets = [[[], []] for _ in range(numColors)]
    for y, row in enumerate(board):
        for x, (val, isCircle) in enumerate(row):
            if isCircle:
                if len(sets[val][0]) == 0:
                    sets[val][0].append((x, y))
                else:
                    sets[val][1].append((x, y))
    solutionBoard, solutionSets = SearchForSolution(board, sets)
    return solutionBoard, solutionSets


def SearchForSolution(board, sets):
    if IsValidSolution(board, sets):
        return board, sets
    options = GetAllOptions(board, sets)
    options.sort(key=lambda x: x[-1] * 1000 + x[-1])
    for option in options:
        nextBoard, nextSets = Move(board, sets, option[0], option[1], option[2])
        solution, solutionSets = SearchForSolution(nextBoard, nextSets)
        if solution is not None:
            return solution, solutionSets
    return None, None


def GetAllOptions(board, sets):
    options = []
    for colorInd, set in enumerate(sets):
        firstSet = set[0]
        secondSet = set[1]
        if NextTo(firstSet[-1], secondSet[-1]):
            continue
        firstOptions = GetNeighbors(board, firstSet[-1])
        for option in firstOptions:
            manhattanDistance = Minus(option, secondSet[-1])
            manhattanDistance = abs(manhattanDistance[0]) + abs(manhattanDistance[1])
            options.append((colorInd, 0, option, manhattanDistance, len(firstOptions)))
        secondOptions = GetNeighbors(board, secondSet[-1])
        for option in secondOptions:
            manhattanDistance = Minus(option, firstSet[-1])
            manhattanDistance = abs(manhattanDistance[0]) + abs(manhattanDistance[1])
            options.append((colorInd, 1, option, manhattanDistance, len(secondOptions)))
    return options


def GetNeighbors(board, position):
    options = []
    offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for offset in offsets:
        option = Minus(position, offset)
        if not OnBoard(board, option):
            continue
        options.append(option)
    return options


def Move(board, sets, colorInd, setInd, moveTo):
    nextBoard = [row[:] for row in board]
    nextSet = [[thisSet[0][:], thisSet[1][:]] for thisSet in sets]
    moveFrom = nextSet[colorInd][setInd][-1]
    assert (OnBoard(board, moveTo))
    assert NextTo(moveFrom, moveTo), f'{moveFrom} is not next to {moveTo}'
    assert nextBoard[moveFrom[1]][moveFrom[0]][0] == colorInd
    assert nextBoard[moveTo[1]][moveTo[0]][0] == -1
    nextSet[colorInd][setInd].append(moveTo)
    nextBoard[moveTo[1]][moveTo[0]] = (colorInd, False)
    return nextBoard, nextSet


def IsValidSolution(board, sets):
    # Check that we have a connection between each circle
    # and that each position is correct in the board
    for colorId, set in enumerate(sets):
        if not NextTo(set[0][-1], set[1][-1]):
            return False
        for i in range(0, 2):
            for pos in set[i]:
                # Assert because this means there's a malformed board state
                assert board[pos[1]][pos[0]][0] == colorId

    # Check that each position on the board is filled
    for row in board:
        for val in row:
            if val[0] == -1:
                return False
    return True


def NextTo(pos, otherPos):
    diff = Minus(pos, otherPos)
    return (abs(diff[0]) == 1 and diff[1] == 0) or (diff[0] == 0 and abs(diff[1]) == 1)


def OnBoard(board, pos):
    return pos[1] >= 0 and pos[1] < len(board) and pos[0] >= 0 and pos[0] < len(board[pos[1]])


def Minus(pos, otherPos):
    diff = (pos[0] - otherPos[0], pos[1] - otherPos[1])
    return diff
