import numpy as np
from copy import deepcopy

previousFrame = np.array([['id', 'ob' ,None ,None,'id', None, None, None, None, None, 'ob', 'id'],
                            ['sb', None, None, None, 'a', None, 'sb', None, None, None, None, 'sb'],
                            [None, 'id', 'ob', None, None, None, None, None, None, 'id', 'sb', None],
                            [None, 'sb', None, 'sb', 'sb', 'id', None, 'id', 'id', 'b', None, 'sb'],
                            [None, None, None, None, None, None, None, 'sb', 'id', 'sb', None, 'id'],
                            [None, None, 'sb', 'id', None, 'E' ,None ,'X', None, None, None, None],
                            [None, 'id', None, 'b', None, 'b', None, None, 'ob', None, None, None],
                            [None, 'id', 'sb', None, None, 'a' ,'sb' ,None ,'id', 'id', None, 'id'],
                            [None, 'ob', None, 'a' ,None ,None ,None ,None ,'id', None, None, 'sb'],
                            [None, None, None, None, 'a' ,'sb' ,None ,None ,None, 'sb', 'id', None]],dtype='object')


currentFrame = np.array([['id', 'ob' ,None ,None ,'id', None, None, None, None, None, 'ob', 'id'],
                            ['sb', None, None, None, 'a', None, 'sb', None, None, None, None, 'sb'],
                            [None, 'id', 'ob', None, None, None, None, None, None, 'id', 'sb', None],
                            [None, 'sb', None, 'sb', 'sb', 'id', None, 'id', 'id', None, None, 'sb'],
                            [None, None, None, None, None, None, None, 'sb', 'id', 'sb', None, 'id'],
                            [None, None, 'sb', 'id', None, 'E' ,None ,'X', None, None, None, None],
                            [None, 'id', None, None, None, None, None, None, 'ob', None, None, None],
                            [None, 'id', 'sb', None, None, 'a' ,'sb' ,None ,'id', 'id', None, 'id'],
                            [None, 'ob', None, 'a' ,None ,None ,None ,None ,'id', None, None, 'sb'],
                            [None, None, None, None, 'a' ,'sb' ,None ,None ,None, 'sb', 'id', None]],dtype='object')


def last2FramesSubstraction(previousFrame, currentFrame):
    """
    function that maps the blast radius of bombs and return if the ennemy agent was hit
    """
    explosionList = list()

    for rowidx, row in enumerate(currentFrame):
        for colidx, _ in enumerate(row):

            if (currentFrame[rowidx][colidx] is None) and (previousFrame[rowidx][colidx] == 'b'):
                # need to add x + y of what was touched (+2)
                # also since there will be power ups we need to track who places what bombs
                currentFrame[rowidx][colidx] = 'BOOM'
                print(f'explosion at (y:{9 - rowidx},x:{colidx})')
                explosionList.append((9 - rowidx,colidx))
                #print(explosionList)

            elif (currentFrame[rowidx][colidx] in ['E','X','id']):
                pass

            elif currentFrame[rowidx][colidx] is None and previousFrame[rowidx][colidx]!='b':
                # could be sb,X,E taht moved
                currentFrame[rowidx][colidx] = ''

            elif previousFrame[rowidx][colidx] == currentFrame[rowidx][colidx]:
                #print(currentFrame[rowidx][colidx])
                currentFrame[rowidx][colidx] = currentFrame[rowidx][colidx].replace(previousFrame[rowidx][colidx],'')

    for explosion in explosionList:

        # y boundaries
        minRow = 0
        maxRow = 9
        # x boundaries
        minCol = 0
        maxCol = 11
        # recap: done this way because of how numpy arrays are laid out
        # also we need to inverse row indexing so that's why we use maxrow - row
        # frame[row or y][col or x]

        # row edge cases (y coord)
        if explosion[0] - 3 < 0:
            minRow = 0
        else:
            minRow = explosion[0]-3
        
        if explosion[0] + 3 > 9:
            # 10 instead of 9 (the actual limit) because range ignores the last value
            maxRow = 10
        else:
            maxRow = explosion[0] + 3
        
        # columns edge cases (x coord)
        if explosion[1] - 2 < 0:
            minCol = 0
        else:
            minCol = explosion[1]-2
        
        if explosion[1] + 3 > 11:
            # 12 instead of 11 (the actual limit) because range ignores the last value
            maxCol = 12
        else:
            maxCol = explosion[1] + 3

        #print(minCol,explosion[1],maxCol)
        #print(minRow,explosion[0],maxRow)
        
        # mapping the rows blast
        for x in range(explosion[0], maxRow):

            # there's a wall at the above the explosion
            if (currentFrame[9-x][explosion[1]] == 'id'): 
                break

            # empty space so the explosion propagate
            elif currentFrame[9-x][explosion[1]] == '': 
                currentFrame[9-x][explosion[1]] = 'bl'
            # there's an player that was touched by the explosion
            else:
                if currentFrame[9-x][explosion[1]][0]=='[' and currentFrame[9-x][explosion[1]][-1] ==']':
                    pass
                else:
                    currentFrame[9-x][explosion[1]] = '['+currentFrame[9-x][explosion[1]]+']'

        for x in range(explosion[0], minRow,-1):

            # there's a wall at bellow explosion
            if (currentFrame[9-x][explosion[1]] == 'id'): 
                break

            # empty space so the explosion propagate
            elif currentFrame[9-x][explosion[1]] == '': 
                currentFrame[9-x][explosion[1]] = 'bl'
            # there's an player that was touched by the explosion
            else:
                if currentFrame[9-x][explosion[1]][0]=='[' and currentFrame[9-x][explosion[1]][-1] ==']':
                    pass
                else:
                    currentFrame[9-x][explosion[1]] = '['+currentFrame[9-x][explosion[1]]+']'

        # mapping the columns blast
        for y in range(explosion[1], maxCol):
            # there's a wall at the right of the explosion
            if currentFrame[9-explosion[0]][y] == 'id':
                print('gtfo',currentFrame[9-explosion[0]][y] )
                break
            # empty space so the explosion propagate
            elif currentFrame[9-explosion[0]][y] == '': 
                currentFrame[9-explosion[0]][y] = 'bl'
            # there's an player that was touched by the explosion
            else:
                if currentFrame[9-explosion[0]][y][0] == '[' and currentFrame[9-explosion[0]][y][-1] == ']':
                    pass
                else:
                    currentFrame[9-explosion[0]][y] = '['+currentFrame[9-explosion[0]][y]+']'

        for y in range(explosion[1], minCol,-1):
            if currentFrame[9-explosion[0]][y] == 'id':
                break
            # empty space so the explosion propagate
            elif currentFrame[9-explosion[0]][y] == '': 
                currentFrame[9-explosion[0]][y] = 'bl'
            # there's an player that was touched by the explosion
            else:
                if currentFrame[9-explosion[0]][y][0] == '[' and currentFrame[9-explosion[0]][y][-1] == ']':
                    pass
                else:
                    currentFrame[9-explosion[0]][y] = '['+currentFrame[9-explosion[0]][y]+']'
    ennemyAgentHit = np.where(currentFrame == '[E]')
    if np.size(ennemyAgentHit):
        print('ennemy hit')
last2FramesSubstraction(previousFrame,currentFrame)
print(currentFrame)
