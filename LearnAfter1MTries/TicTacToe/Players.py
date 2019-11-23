import pygame as pg
from Settings import *
import random


class Player(pg.sprite.Sprite):
    def __init__(self, environ,type,order, x, y):
        self.groups = environ.all_sprites, environ.players
        pg.sprite.Sprite.__init__(self, self.groups)
        self.environ = environ
        self.image = pg.Surface((TILESIZE*2, TILESIZE*2))
        self.type= type
        self.order=order
        self.move =[]
        if self.order == PLAYER1:
            self.image.fill(RED)
        if self.order == PLAYER2:
            self.image.fill(GREEN)

        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE-TILESIZE/2
        self.rect.y = y * TILESIZE-TILESIZE/2
        self.state = LEARNING
        self.playCount=0
        self.playerMoves = []

    def RandomPlay(self):
        moves=self.environ.moves
        if moves !=[]:
            self.move=random.choice(moves)
            self.environ.moves.remove(self.move)
        #print("Player {} move {}".format(str(self.order),str(self.move)))

    def openlogFile(self,fileName):
        with open(fileName, "a") as file:
            file.write(self.move.append(self.fitness))

    def log(self,fileName,fitness):
        with open(fileName, "a") as file:
            logMoves=self.playerMoves
            stringToWrite=''
            for index,moves in enumerate(logMoves):
                    stringToWrite=stringToWrite+moves+','

            if fitness ==1:
                addString="Player1"
            elif fitness ==-1:
                addString = "Player2"
            elif fitness == 0:
                addString = "Draw"
            else:
                addString= "Invalid"

            stringToWrite = stringToWrite + addString
            #print(stringToWrite)
            file.write(stringToWrite)
            file.write('\n')

    def play(self):
        if self.type ==PLAYER_TYPE_BAYES:
            if self.state==LEARNING:
                self.RandomPlay()
                self.playCount +=1
                if self.playCount ==LEARN_NUM:
                    self.state == TRAINED
                    self.playCount = 0
            else:
                self.BayesPlay()

        return self.move

    def BayesPlay(self):
        self.RandomPlay()
        #print("Player {} move {}".format(str(self.order),str(self.move)))

    def GANNPlay(self):
        pass

    def setFitness(self,fitness):
        pass





