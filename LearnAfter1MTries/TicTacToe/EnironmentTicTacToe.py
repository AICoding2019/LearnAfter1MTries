#!/usr/bin/env python
import pygame as pg
import numpy as np
import os.path
import sys
from os import path
from Settings import *
from Map import *
from Walls import *
from Players import *
from EventManager import *

# see if we can load more than standard BMP
if not pg.image.get_extended():
    raise SystemExit("Sorry, extended image module required")


class TicTacToe:
    def __init__(self, player1Type, player2Type):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        self.surface = pg.Surface(self.screen.get_size(), pg.SRCALPHA, 32)
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.map = []
        self.loadImageData()
        self.locDisplay = {(0, 0): (2, 2),
                           (0, 1): (8, 2),
                           (0, 2): (14, 2),
                           (1, 0): (2, 6),
                           (1, 1): (8, 6),
                           (1, 2): (14, 6),
                           (2, 0): (2, 10),
                           (2, 1): (8, 10),
                           (2, 2): (14, 10),
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

        MaskPosSecRow = np.array([[0, 0, 0],
                                  [1, 1, 1],
                                  [0, 0, 0]])

        MaskPosThirdRow = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [1, 1, 1]])

        MaskPosFirstCol = np.array([[1, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 0]])

        MaskPosSecCol = np.array([[0, 1, 0],
                                  [0, 1, 0],
                                  [0, 1, 0]])

        MaskPosThirdCol = np.array([[0, 0, 1],
                                    [0, 0, 1],
                                    [0, 0, 1]])

        MaskPosLDiagRow = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

        MaskPosRDiagRow = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]])

        self.winPositionMask = [MaskPosFirstRow, MaskPosSecRow,
                                MaskPosThirdRow, MaskPosFirstCol,
                                MaskPosSecCol, MaskPosThirdCol,
                                MaskPosLDiagRow, MaskPosRDiagRow]

        self.moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.movesLeft = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.playerMoves = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
        self.opponentMoves = []
        self.board = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])

        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.players = pg.sprite.Group()

        self.player1 = Player(self, player1Type, PLAYER1, 0, 0)
        self.player2 = Player(self, player2Type, PLAYER2, 0, 0)
        self.events = EventManager(self)

        self.map

    def loadImageData(self):
        ticTacToeFolder = os.path.dirname(__file__)
        self.map = Map(os.path.join(ticTacToeFolder, 'map.txt'))

    def boardDisplay(self):
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self, col, row)

        for row in range(0, 3):
            for col in range(0, 3):
                if self.board[row, col] == 1:
                    displayPoint = self.locDisplay[(row, col)]
                    Player(self, PLAYER_TYPE_BAYES, PLAYER1, displayPoint[0], displayPoint[1])

                if self.board[row, col] == 2:
                    displayPoint = self.locDisplay[(row, col)]
                    Player(self, PLAYER_TYPE_BAYES, PLAYER2, displayPoint[0], displayPoint[1])

    def newSim(self):
        self.player1.move = []
        self.player2.move = []
        self.playerMoves = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
        self.board = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        self.moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.killAllSprite()

    def killAllSprite(self):
        for sprite in self.all_sprites:
            sprite.kill()

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.all_sprites.draw(self.screen)
        self.screen.blit(self.surface, (0, 0))
        pg.display.flip()
        dirty = self.all_sprites.draw(self.screen)
        pg.display.update(dirty)
        pg.time.wait(1000)
        # draw the scene


    def laws(self):
        PlayerStatus, mask = self.checkBoardIfWin()
        # print("PlayerStatus Law  {} moves left {}:".format(PlayerStatus,self.moves))

        if PlayerStatus != "UNKNOWN":
            status = PlayerStatus
            return status

        if PlayerStatus == "UNKNOWN" and self.movesLeft != []:
            status = "OPEN"

        if (status != "WIN") and (status != "LOSE") and (self.moves == []):
            status = "DRAW"

        return status

    def checkBoardIfWin(self):
        status = "UNKNOWN"
        winMask = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])

        for mask in self.winPositionMask:
            winOrLose = self.board * mask

            if np.array_equal(winOrLose, mask):
                status = "WIN"
                return status, mask

            if np.array_equal(winOrLose / 2, mask):
                status = "LOSE"
                return status, mask

        return status, winMask

    def playMove(self, desiredMove, player):
        status = self.laws()

        if self.board[desiredMove] == 0:
            self.board[desiredMove] = player
            status = self.laws()
            if self.board[desiredMove] == []:
                return status
        else:
            if self.board[desiredMove] != []:
                status = "INVALID"

        return status

    def fitnessScore(self, status):
        score = "UNKNOWN"
        if status == "WIN":
            score = 1
        elif status == "DRAW":
            score = 0
        elif status == "LOSE":
            score = -1
        elif status == "INVALID":
            score = -2
        else:
            pass

        return score

    def update(self):
        pass

    def play(self):
        player1Move = self.player1.play()
        status = self.playMove(player1Move, 1)

        if status != "INVALID":
            dataCell = self.locData[player1Move]
            self.playerMoves[dataCell] = '1'

            if status != "WIN":
                player2Move = self.player2.play()
                status = self.playMove(player2Move, 2)

                if status != "INVALID":
                    dataCell = self.locData[player2Move]
                    self.playerMoves[dataCell] = '2'

        return status


if __name__ == '__main__':
    environ = TicTacToe(PLAYER_TYPE_BAYES, PLAYER_TYPE_BAYES)

    running = True
    count = 0
    countInvalid = 0
    countValid = 0
    countWin = 0
    countLose = 0
    countDraw = 0

    while running:
        status = environ.play()

        count += 1

        environ.player1.playerMoves = environ.playerMoves
        fitness = environ.fitnessScore(status)

        if status == "WIN":
            countWin += 1
        if status == "LOSE":
            countLose += 1
        if status == "DRAW":
            countDraw += 1

        if status == "WIN" or status == "LOSE" or status == "DRAW":
            # print("fitnessScore : {}".format(fitness))
            # print(status)
            # print(environ.board)
            print("Games-{}-Valid-{}-Invaild-{}-Win-{}-Lose-{}-Draw-{}".format(countValid + countInvalid, countValid,
                                                                               countInvalid,
                                                                               countWin, countLose, countDraw))

            environ.player1.setFitness(fitness)
            environ.player2.setFitness(fitness)
            environ.player1.log("gameLog.txt", fitness)
            countValid += 1
            #environ.newSim()

        if status == "INVALID":
            environ.player1.setFitness(fitness)
            environ.player2.setFitness(fitness)
            environ.newSim()
            countInvalid += 1

        if (countValid + countInvalid) == 1:
            running = False

        # print(environ.board)
        environ.boardDisplay()
        environ.draw()
        environ.clock.tick(FPS)

    while True:
        environ.events.events()
    print("SIM END")
