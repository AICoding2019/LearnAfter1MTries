import pygame as pg
from Settings import *


class Wall(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self.groups = environ.all_sprites, environ.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE