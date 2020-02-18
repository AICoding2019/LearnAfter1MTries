import pygame as pg
from LearnAfter1MTries.EventManager.EventManager import EventManager

# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (200, 200, 200)
DISABLEGREY = (240, 240, 240)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (0, 255, 255)


class GraphWindow:
    def __init__(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.perceptron = pg.sprite.Group()
        self.screen = pg.display.set_mode((600, 600))
        pg.display.set_caption("Test NN render")
        self.clock = pg.time.Clock()
        pg.init()
        self.font = pg.font.Font(pg.font.get_default_font(), 20)
        self.renderText = self.font.render("Start", True, (0, 128, 0))

    def draw(self):
        self.screen.fill(WHITE)
        # self.screen.blit("NN", (10, 700))
        self.all_sprites.draw(self.screen)
        pg.display.flip()
        self.clock.tick(1)

    def eventManager(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.quit()


class Perceptron(pg.sprite.Sprite):
    def __init__(self, environ, x=0, y=0,Enabled=True, Activation=True, Recurrent=True, TILESIZE=64, XOFFSET=0, YOFFSET=0):
        self._layer = 3
        self.groups = environ.all_sprites, environ.perceptron
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(WHITE)
        #pg.draw.circle(self.image, MAGENTA, (32, 32), 32)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = (x + XOFFSET) * TILESIZE
        self.rect.y = (y + YOFFSET) * TILESIZE

        if Recurrent:
            self.image.fill(LIGHTGREY)

        if not Enabled:
            pg.draw.circle(self.image, DISABLEGREY, (32, 32), 32)
        elif Activation== 'sigmoid':
            pg.draw.circle(self.image, RED, (32, 32), 32)
        elif Activation == 'relu':
            pg.draw.circle(self.image, GREEN, (32, 32), 32)
        elif Activation == 'leakyRelu':
            pg.draw.circle(self.image, BLUE, (32, 32), 32)
        elif Activation == 'tanh':
            pg.draw.circle(self.image, YELLOW, (32, 32), 32)
        elif Activation == 'linear':
            pg.draw.circle(self.image, DARKGREY, (32, 32), 32)


class Inputs(pg.sprite.Sprite):
    def __init__(self, environ, x, y, TILESIZE, XOFFSET, YOFFSET):
        self._layer = 3
        self.groups = environ.all_sprites, environ.perceptron
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(WHITE)
        pg.draw.circle(self.image, MAGENTA, (32, 32), 32)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = (x + XOFFSET) * TILESIZE
        self.rect.y = (y + YOFFSET) * TILESIZE


class Outputs(pg.sprite.Sprite):
    def __init__(self, environ, x, y, TILESIZE, XOFFSET, YOFFSET):
        self._layer = 3
        self.groups = environ.all_sprites, environ.perceptron
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(WHITE)
        pg.draw.circle(self.image, MAGENTA, (32, 32), 32)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = (x + XOFFSET) * TILESIZE
        self.rect.y = (y + YOFFSET) * TILESIZE


if __name__ == '__main__':
    Environ = GraphWindow()
    testNode1 = Perceptron(Environ)

    Environ.draw()

    # pg.display.update()
