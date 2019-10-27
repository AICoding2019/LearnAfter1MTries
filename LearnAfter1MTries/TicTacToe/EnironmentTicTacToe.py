import pygame as pg
import numpy as np
import sys
from os import path
from Settings import *
from Map import *
from Walls import *
from Players import *


class TicTacToe:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH,HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock =pg.time.Clock()
        self.loadImageData()
        self.locDisplay={(0,  0): (2,  2),
                         (0,  1): (8,  2),
                         (0,  2): (14, 2),
                         (1,  0): (2,  6),
                         (1,  1): (8,  6),
                         (1,  2): (14, 6),
                         (2,  0): (2,  10),
                         (2,  1): (8,  10),
                         (2,  2): (14, 10),
                        }

        self.locData = {(0, 0): 0,
                        (0, 1): 1,
                        (0, 2): 2,
                        (1, 0): 3,
                        (1, 1): 4,
                        (1, 2): 5,
                        (2, 0): 6,
                        (2, 1): 7,
                        (2, 2): 8,
                        }


        MaskPosFirstRow = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [0, 0, 0]])

        MaskPosSecRow   = np.array([[0, 0, 0],
                                    [1, 1, 1],
                                    [0, 0, 0]])

        MaskPosThirdRow = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [1, 1, 1]])

        MaskPosFirstCol = np.array([[1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0]])

        MaskPosSecCol =   np.array([[0, 1, 0],
                                    [0, 1, 0],
                                    [0, 1, 0]])

        MaskPosThirdCol = np.array([[0, 0, 1],
                                    [0, 0, 1],
                                    [0, 0, 1]])

        MaskPosLDiagRow = np.array([[1, 0, 0],
                                    [0, 1,  0],
                                    [0, 0, 1]])

        MaskPosRDiagRow = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]])

        self.winPositionMask=[MaskPosFirstRow,MaskPosSecRow,
                              MaskPosThirdRow,MaskPosFirstCol,
                              MaskPosSecCol,MaskPosThirdCol,
                              MaskPosLDiagRow,MaskPosRDiagRow]

        self.moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.movesLeft=[0,1,2,3,4,5,6,7,8]
        self.playerMoves=[0,0,0,0,0,0,0,0,0]
        self.opponentMoves=[]
        self.board =np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]])

        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.players = pg.sprite.Group()

        self.player1 = Player(self, PLAYER_TYPE_BAYES, PLAYER1,0,0)
        self.player2 = Player(self, PLAYER_TYPE_BAYES, PLAYER2,0,0)


    def loadImageData(self):
        ticTacToeFolder=path.dirname(__file__)
        self.map =Map(path.join(ticTacToeFolder, 'map.txt'))

    def boardDisplay(self):

        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self,col,row)
               
        for row in range(0,3):
            for col in range(0,3):
                if self.board[row,col] ==1:
                    displayPoint=self.locDisplay[(row,col)]
                    Player(self, PLAYER_TYPE_BAYES, PLAYER1, displayPoint[0], displayPoint[1])


                if self.board[row, col] == 2:
                    displayPoint = self.locDisplay[(row,col)]
                    Player(self, PLAYER_TYPE_BAYES, PLAYER2, displayPoint[0], displayPoint[1])

    def newSim(self):
        player1.move=[]
        player2.move=[]


    def draw(self):
        self.screen.fill(BGCOLOR)
        self.all_sprites.draw(self.screen)
        pg.display.flip()


    def laws(self,playerdesiredMove):

        #Players move that already played
        print("------{},{}".format(playerdesiredMove,self.board[playerdesiredMove]))


        PlayerStatus,mask =self.checkIfWin()
        print("PlayerStatus :"+ PlayerStatus)

        if PlayerStatus !="UNKNOWN":
            status = PlayerStatus
            return status
        elif PlayerStatus =="UNKNOWN" and self.movesLeft !=[]:
            status="OPEN"

        if (status != "WIN") and (status != "LOSE") and (self.movesLeft ==[]):
            status ="DRAW"

        return status

    def checkIfWin(self):
        status = "UNKNOWN"
        winMask=np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        for mask in self.winPositionMask:

            winOrLose=self.board*mask

            if np.array_equal(winOrLose , mask):
                status="WIN"
                return status , mask

            if np.array_equal(winOrLose/2 , mask):
                status="LOSE"
                return status , mask

        return status, winMask

    def playMove (self,desiredMove, player):

        if self.board[desiredMove] == 0:
            self.board[desiredMove]=player
            status=self.laws(desiredMove)
        else:
            status = "INVALID"

       # print("-----" +str(player)+str(desiredMove))
       # print(self.board)
       # print("-----" +str(player))
       # print(self.laws(desiredMove))
        return status



    def fitnessScore(self,status):
        if status =="WIN":
            score=1
        elif status =="DRAW":
            score=-1
        elif status =="LOSE":
            score=0
        elif status =="INVALID":
            score=-2
        else:
            pass

        return score

    #def environState(se   lf,board):
     #   pass

    def update(self):
        pass

    def play(self):

        player1Move=self.player1.play()
        status=self.playMove(player1Move, 1)
        print("---------------------------{},status".format(status))

        if status != "INVALID":
            dataCell = self.locData[player1Move]
            self.playerMoves[dataCell]=1


        player2Move = self.player2.play()
        status = self.playMove(player2Move, 2)

        if status != "INVALID":
            dataCell = self.locData[player2Move]
            self.playerMoves[dataCell] = 2


        return status



if __name__ == '__main__':
    environ=TicTacToe()
    print(environ.board)
    print('-------')

    running=True
    count =0
    while running:
        environ.play()

        print(environ.board)
        environ.boardDisplay()
        environ.draw()
        environ.clock.tick(FPS)
        count += 1
        print("----------"+str(count))

        print(environ.playerMoves)
        #fitness=environ.fitnessScore(play)
        #print(fitness)
        # environ.player1.log(fitness)
        if count ==9:
            running=False

    while True:
        pass

    



