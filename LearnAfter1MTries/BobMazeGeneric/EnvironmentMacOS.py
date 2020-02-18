#!/usr/bin/env python
'''
Another version of asteroid game, this time using the pygame Sprite class.
'''
import pygame
from random import randint
import math
from LearnAfter1MTries.TWEANN.GeneticAlgorithm import GA

# Initialize
pygame.init()

# Define a frame rate
fps = 1000

# Define a constant for some color triples (red, green, blue)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
LIGHTGREY = (100, 100, 100)

# Some other fixed game attributes
VELOCITY_SCALE = 12
GRAVITY = 0.2
BASE_X = 300
BASE_Y = 400
TURRET_LENGTH = 25
PROJECTILE_SIZE = 20
ENEMY_VELOCITY = 1
BOB_SIZE = int(PROJECTILE_SIZE / 2)


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((PROJECTILE_SIZE, PROJECTILE_SIZE))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * PROJECTILE_SIZE
        self.rect.y = y * PROJECTILE_SIZE


class EndPoint(pygame.sprite.Sprite):
    def __init__(self, environ, x, y, fittest=True):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((PROJECTILE_SIZE, PROJECTILE_SIZE))
        if fittest:
            self.image.fill(BLUE)
        else:
            self.image.fill(LIGHTGREY)

        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * PROJECTILE_SIZE
        self.rect.y = y * PROJECTILE_SIZE


class Maze:
    def __init__(self):
        self._layer = 3
        self.all_sprites = pygame.sprite.Group()
        self.bob = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.endPoints = pygame.sprite.Group()
        self.BobStartX = 0
        self.BobStartY = 0
        self.ExitX = 0
        self.ExitY = 0
        self.EndPointCount = 0
        self.MapDisplay()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.renderText = self.font.render("Start", True, (0, 128, 0))

        self.Bob = Bob(self.all_sprites, self.bob, self.walls, self.BobStartX, self.BobStartY)
        self.moves = []
        self.moveIndex = 0

        self.size = (600, 400)
        self.all_sprites.add(self.bob)

        # Create a window that we can draw on
        self.screen = pygame.display.set_mode(self.size)

        # Optionally, set a caption
        pygame.display.set_caption("Bob Maze TWEANN")

        # Create a Clock to control the frame rate
        self.clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self.done = False

    def update(self):
        self.TestRoute()

    def loadImageData(self):
        with open('mazemap.txt', "rt") as file:
            data = [line.strip() for line in file]

        return data

    def MapDisplay(self):
        data = self.loadImageData()
        for row, tiles in enumerate(data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Walls = Wall(col, row)
                    self.all_sprites.add(Walls)
                    self.walls.add(Walls)
                if tile == '8':
                    self.BobStartX = col
                    self.BobStartY = row
                # print("{},{}".format(col, row))
                if tile == '5':
                    self.ExitX = col
                    self.ExitY = row
                    # print("{},{}".format(col, row))

    def TestRoute(self, directions=['East', 'East', 'East'], reset=1):
        self.Bob.x = self.BobStartX
        self.Bob.y = self.BobStartY

        for numSteps,direction in enumerate(directions):
            self.Bob.move(direction)
            self.draw()
            if self.Bob.x == self.ExitX and self.Bob.y == self.ExitY:
                break

        dist = math.sqrt((self.Bob.x - self.ExitX) ** 2 + (self.Bob.y - self.ExitY) ** 2)
        invDist = 1 / (dist + 1) + 1/numSteps
        endPoint = (self.Bob.x, self.Bob.y, numSteps)

        # print(f"EndPoint: {invDist} {endPoint} {dist} {(self.ExitX,self.ExitY)}")

        # self.reset()
        return invDist, endPoint

    def showMoves(self):
        self.rect.x = self.moves[self.moveIndex][0]
        self.rect.y = self.moves[self.moveIndex][1]

        self.moveIndex += 1
        if self.moveIndex == len(self.moves):
            self.moveIndex = 0

    # def reset(self):
    def Events(self):
        # Loop through events and update values accordingly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.Bob.move('North')
                if event.key == pygame.K_DOWN:
                    self.Bob.move('South')
                if event.key == pygame.K_RIGHT:
                    self.Bob.move('East')
                if event.key == pygame.K_LEFT:
                    self.Bob.move('West')

    def EndPointDraw(self, endPoint, fittestEver=True):
        if self.EndPointCount > 3:
            self.EndPointCount = 0
            self.all_sprites.remove(self.endPoints)
            print("Too many EndPoints'")
        EndPoints = EndPoint(self, endPoint[0], endPoint[1], fittestEver)
        self.all_sprites.add(EndPoints)
        self.endPoints.add(EndPoints)
        self.EndPointCount += 1

    def draw(self):
        self.Events()
        self.all_sprites.update()
        self.screen.fill(GREY)
        self.all_sprites.draw(self.screen)
        self.screen.blit(self.renderText, (0, 200))
        pygame.display.flip()
        self.clock.tick(fps)

    def infoBoard(self, text):
        self.renderText = self.font.render(text, True, (0, 8, 0))


class Bob(pygame.sprite.Sprite):
    def __init__(self, all_sprites, bob, walls, x, y):
        self._layer = 3
        self.groups = all_sprites, bob
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.walls = walls
        self.image = pygame.Surface((PROJECTILE_SIZE, PROJECTILE_SIZE))
        self.image.fill(GREY)
        pygame.draw.circle(self.image, RED, (BOB_SIZE, BOB_SIZE), BOB_SIZE)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * PROJECTILE_SIZE
        self.rect.y = y * PROJECTILE_SIZE
        self.stopBob = False

    def move(self, direction):
        dx, dy = self.decodeMove(direction)
        # print(f' dx= {dx}, dy= {dy}')
        if not self.Collision(dx, dy):
            self.x += dx
            self.y += dy
            self.rect.x = self.x * PROJECTILE_SIZE
            self.rect.y = self.y * PROJECTILE_SIZE

        # print(direction)
        return self.x, self.y

    def Collision(self, dx=0, dy=0):
        for wall in self.walls:
            if wall.x == self.x + dx and wall.y == self.y + dy:
                return True
        return False

    @staticmethod
    def decodeMove(direction):
        if direction == "North":
            dy = -1
            dx = 0
        elif direction == "South":
            dy = 1
            dx = 0
        elif direction == "East":
            dx = 1
            dy = 0
        else:
            dx = -1
            dy = 0

        return dx, dy


if __name__ == '__main__':
    testEnviron = Maze()
    decodeDict = {0: 'North',
                  1: 'South',
                  2: 'East',
                  3: 'West'}
    # test = GA(PopSize=1000, ChromoLength=64, DecodeDict=decodeDict, numGeneration=100, MutationRate=0.01,
    #           fitnessTest=testEnviron.TestRoute,
    #           selectionType='rouletteWheel',
    #           infoBoard=testEnviron.infoBoard,
    #           progressGen=testEnviron.EndPointDraw,
    #           progressOverall=testEnviron.EndPointDraw)

    test = GA(PopSize=100, numGeneration=100, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=64, GeneLength=2,
              crossOverType='partiallyMapped', mutateType='RealNumber',
              chromoType='RealNumberInt',
              DecodeDict=decodeDict,
              fitnessTest=testEnviron.TestRoute,
              selectionType='rouletteWheel',
              infoBoard=testEnviron.infoBoard,
              progressGen=testEnviron.EndPointDraw,
              progressOverall=testEnviron.EndPointDraw)

    while not testEnviron.done:
        test.Evolve()
        print("Sim Done'")
        testEnviron.infoBoard(f"Gen= {test.Generation} "
                              f"best score ={test.BestFitnessScore:.3f}={test.FittestGenome['Fitness']:.3f}"
                              f" Total fitness={test.TotalFitnessScore:.3f}"
                              f" fitness Info{test.FittestGenome['Info']}"
                              f" fitness Ever Info{test.FittestGenomeEver['Info']}"
                              f" Experiment Completed")
        testEnviron.done = True
        testEnviron.draw()

    while True:
        fps = 1
        directions = test.Decode(test.FittestGenomeEver['chromo'])
        testEnviron.TestRoute(directions, False)
        testEnviron.draw()
        # testEnviron.reset()
#            testEnviron.events.events()
# clean up after loop exits
pygame.quit()
