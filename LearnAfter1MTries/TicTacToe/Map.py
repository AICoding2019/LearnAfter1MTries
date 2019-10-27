import pygame as pg
from Settings import *

class Map:
    def __init__(self, filename):
        with open(filename, "rt") as file:
            self.data=[line.strip() for line in file]

        self.tilewidth = len(self.data[0])
        self.tileheight = len(self.data)
        self.width = self.tilewidth * TILESIZE
        self.height = self.tileheight * TILESIZE

class Objects(pg.sprite.Sprite):
    def __init__(self,environ,image,groups,x,y):
        self.groups=environ.groups
        pg.sprite.Sprite.__init__(self,self.groups)
        self.eviron=environ
        self.image=image
        self.rect=self.image.get_rect()
        self.x=x
        self.y=y
        self.rect.x=x*TILESIZE
        self.rect.y=y*TILESIZE
