import pygame as pg
import math
from LearnAfter1MTries.NEAT.GeneticAlgorithm import GA
from LearnAfter1MTries.EventManager.EventManager import EventManager

# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# game settings
WIDTH = 1540  # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 1040  # 16 * 48 or 32 * 24 or 64 * 12
FPS = 10000
TITLE = "GA Maze Solver"
BGCOLOR = WHITE

TILESIZE = 64
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE


class Environment:
    def __init__(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.walls = pg.sprite.Group()
        self.trails = pg.sprite.Group()
        self.exits = pg.sprite.Group()
        self.endPoints = pg.sprite.Group()
        self.wallList = []
        self.trailsList = []
        self.bob = pg.sprite.Group()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.data = None
        self.loadImageData()
        self.tileWidth = len(self.data[0])
        self.tileHeight = len(self.data)
        self.width = self.tileWidth * TILESIZE
        self.height = self.tileHeight * TILESIZE
        self.stopBob = False
        self.BobStartX = 0
        self.BobStartY = 0
        self.Route = 0
        self.Fitness = 0
        pg.init()
        self.font = pg.font.Font(pg.font.get_default_font(), 20)
        self.renderText = self.font.render("Start", True, (0, 128, 0))
        self.EndPointCount = 0

        self.BobTrail = None
        self.Bob = None
        self.Exit = None

        self.MapDisplay()
        self.events = EventManager(self)

    def TestRoute(self, directions=[0], reset=1):
        self.Bob.x = self.BobStartX
        self.Bob.y = self.BobStartY
        old_x = self.BobStartX
        old_y = self.BobStartY
        startPoint = (self.Bob.x, self.Bob.y)

        for direction in directions:
            x, y = self.Bob.move(direction)

            self.Bob.Collision()
            if self.Bob.stopBob:
                self.Bob.x, self.Bob.y = (old_x, old_y)
                break
            old_x, old_y = x, y

            self.trailsList.append((x, y))
            self.TrailDraw()
            self.draw()

        dist = math.sqrt((self.Bob.x - self.Exit.x) ** 2 + (self.Bob.y - self.Exit.y) ** 2)
        invDist = 1 / (dist + 1)
        self.Route = directions
        self.Fitness = invDist
        endPoint = (self.Bob.x, self.Bob.y)

        # print("Exit {}, Bob {}, dist {} EndPoint{} {}".format((self.Exit.x,self.Exit.y),
        # (self.Bob.x,self.Bob.y),dist,endPoint,directions))
        if reset:
            self.reset()

        return invDist, endPoint

    def drawMaze(self):
        self.MapDisplay()
        self.draw()

    def reset(self):
        self.Bob.kill()
        self.trailsList = []
        self.all_sprites.remove(self.bob)
        self.all_sprites.remove(self.trails)
        self.all_sprites.remove(self.walls)
        self.all_sprites.remove(self.exits)
        self.trails.remove(self.trails)
        self.walls.remove(self.walls)
        self.MapDisplay()
        self.draw()

    def MapDisplay(self):
        for row, tiles in enumerate(self.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self, col, row)

                if tile == '8':
                    self.BobTrail = Trail(self, col, row)
                    self.Bob = Bob(self, col, row)
                    self.bob.add(self.Bob)
                    self.BobStartX = col
                    self.BobStartY = row
                    # print("{},{}".format(col,row))
                if tile == '5':
                    self.Exit = Exit(self, col, row)
                    # print("{},{}".format(col, row))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.screen.blit(self.renderText, (10, 700))
        self.all_sprites.draw(self.screen)
        pg.display.flip()
        self.clock.tick(FPS)

    def loadImageData(self):
        with open('mazemap.txt', "rt") as file:
            self.data = [line.strip() for line in file]

    def TrailDraw(self):
        for trail in self.trailsList:
            # print("Trail{}, {}".format(trail[0], trail[1]))
            Trail(self, trail[0], trail[1])

    def EndPointDraw(self, endPoint, fittestEver=True):
        if self.EndPointCount > 3:
            self.EndPointCount = 0
            self.all_sprites.remove(self.endPoints)
            print("Too many EndPoints'")
        EndPoint(self, endPoint[0], endPoint[1], fittestEver)
        self.EndPointCount += 1

        self.draw()

    def infoBoard(self, text):
        self.renderText = self.font.render(text, True, (0, 128, 0))


class Exit(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self._layer = 0
        self.groups = environ.all_sprites, environ.exits
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(GREEN)

        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Wall(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self._layer = 0
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
        # self._layer = self.rect.bottom


class Bob(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self._layer = 3
        self.groups = environ.all_sprites, environ.bob
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(WHITE)
        pg.draw.circle(self.image, RED, (32, 32), 32)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
        self.stopBob = False

    def Collision(self):
        bobCollide = pg.sprite.spritecollide(self, self.environ.walls, False)
        if bobCollide:
            self.stopBob = True
            # print("bobCollide {} , stopBob {}, Bobpos {}".format(bobCollide,self.stopBob,(self.x,self.y)))

    def move(self, direction):
        if direction == "North":
            self.y -= 1
            self.rect.y = self.y * TILESIZE

        elif direction == "South":
            self.y += 1
            self.rect.y = self.y * TILESIZE
        elif direction == "East":
            self.x += 1
            self.rect.x = self.x * TILESIZE
        else:
            self.x -= 1
            self.rect.x = self.x * TILESIZE

        # print("direction {}, x {}, y {}".format(direction,self.x,self.y))
        return self.x, self.y


class Trail(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self._layer = 1
        self.groups = environ.all_sprites, environ.trails
        pg.sprite.Sprite.__init__(self, self.groups)
        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE

        # self._layer = self.rect.bottom


class EndPoint(pg.sprite.Sprite):

    def __init__(self, environ, x, y, fittest=True):
        self._layer = 4
        self.groups = environ.all_sprites, environ.endPoints
        pg.sprite.Sprite.__init__(self, self.groups)
        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        if fittest:
            self.image.fill(BLUE)
        else:
            self.image.fill(LIGHTGREY)

        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


if __name__ == '__main__':
    testEnviron = Environment()
    decodeDict = {0: 'North',
                  1: 'South',
                  2: 'East',
                  3: 'West'}
    test = GA(PopSize=1, ChromoLength=64, DecodeDict=decodeDict, numGeneration=1, fitnessTest=testEnviron.TestRoute,
              selectionType='rouletteWheel',
              infoBoard=testEnviron.infoBoard,
              progressGen=testEnviron.EndPointDraw,
              progressOverall=testEnviron.EndPointDraw)
    test.Evolve()
    print("Sim Done'")
    testEnviron.infoBoard(f"Generation= {test.Generation} "
                  f"best fitness score ={test.BestFitnessScore:.3f}={test.FittestGenome['Fitness']:.3f}"
                  f" Total fitness={test.TotalFitnessScore:.3f}"
                  f" Experiment Completed")

    while True:
        FPS = 1
        directions = test.Decode(test.FittestGenomeEver['chromo'])
        testEnviron.TestRoute(directions, False)
        testEnviron.reset()
        testEnviron.events.events()
