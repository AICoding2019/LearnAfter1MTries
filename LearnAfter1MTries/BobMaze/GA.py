import random as rd
from random import random
from random import randint
import pygame as pg
import math
from os import path
# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE= (0, 0, 255)
YELLOW = (255, 255, 0)

# game settings
WIDTH = 1540   # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 1040  # 16 * 48 or 32 * 24 or 64 * 12
FPS = 10
TITLE = "GA Maze Solver"
BGCOLOR = WHITE

TILESIZE = 64
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE


class Map:
    def __init__(self):
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.walls = pg.sprite.Group()
        self.trails = pg.sprite.Group()
        self.exits = pg.sprite.Group()
        self.endPoints = pg.sprite.Group()
        self.wallList =[]
        self.trailsList=[]
        self.bob = pg.sprite.Group()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.loadImageData()
        self.tilewidth = len(self.data[0])
        self.tileheight = len(self.data)
        self.width = self.tilewidth * TILESIZE
        self.height = self.tileheight * TILESIZE
        self.stopBob = False
        self.BobStartX = 0
        self.BobStartY = 0
        self.Route = 0
        self.Fitness = 0
        pg.init()
        self.font = pg.font.Font(pg.font.get_default_font(), 20)
        self.renderText=self.font.render("Start", True, (0, 128, 0))
        self.EndPointCount =0

    def TestRoute(self,directions,reset):
        self.Bob.x=self.BobStartX
        self.Bob.y = self.BobStartY
        old_x=self.BobStartX
        old_y=self.BobStartY
        startPoint=(self.Bob.x,self.Bob.y)

        for dir in directions:
            if dir ==0: #North
                x,y=self.Bob.move("North")
            elif dir ==1: #South
                x,y=self.Bob.move("South")
                #self.trailsList.append((x,y))
            elif dir ==2:#East
                x,y=self.Bob.move("East")
                #self.trailsList.append((x,y))
            else: #WestRED
                x,y=self.Bob.move("West")

            self.Bob.Collison()
            if self.Bob.stopBob == True:
                self.Bob.x,self.Bob.y=(old_x,old_y)
                break
            old_x, old_y=x,y

            self.trailsList.append((x, y))
            self.Traildraw()
            self.draw()

        dist=math.sqrt((self.Bob.x-self.Exit.x)**2+(self.Bob.y-self.Exit.y)**2)
        invDist=1/(dist+1)
        self.Route=directions
        self.Fitness=invDist
        endPoint=(self.Bob.x,self.Bob.y)

        #print("Exit {}, Bob {}, dist {} EndPoint{} {}".format((self.Exit.x,self.Exit.y),(self.Bob.x,self.Bob.y),dist,endPoint,directions))
        if reset:
         self.reset()


        return invDist,endPoint,startPoint,dist

    def drawMaze(self):
        self.MapDisplay()
        self.draw()

    def reset(self):
        self.Bob.kill()
        self.trailsList=[]
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
                    Wall(self,col,row)

                if tile == '8':
                    self.BobTrail = Trail(self, col, row)
                    self.Bob = Bob(self, col, row)
                    self.bob.add(self.Bob)
                    self.BobStartX=col
                    self.BobStartY=row
                    #print("{},{}".format(col,row))
                if tile == '5':
                    self.Exit = Exit(self, col, row)
                    #print("{},{}".format(col, row))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.screen.blit(self.renderText,(10 , 700 ))
        self.all_sprites.draw(self.screen)
        pg.display.flip()
        self.clock.tick(FPS)

    def loadImageData(self):
        with open('mazemap.txt', "rt") as file:
            self.data=[line.strip() for line in file]

    def Traildraw(self):
        for trail in self.trailsList:
            #print("Trail{}, {}".format(trail[0], trail[1]))
            Trail(self, trail[0], trail[1])

    def EndPointdraw(self,endPoint,fittestEver):
        if self.EndPointCount > 3:
            self.EndPointCount = 0
            self.all_sprites.remove(self.endPoints)
            print("Too many EndPoints'")
        EndPoint(self, endPoint[0],endPoint[1],fittestEver)
        self.EndPointCount +=1

        self.draw()


    def infoBoard(self,text):
        self.renderText=self.font.render(text, True, (0, 128, 0))

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
        #self._layer = self.rect.bottom

class Bob(pg.sprite.Sprite):
    def __init__(self, environ, x, y):
        self._layer =2
        self.groups = environ.all_sprites, environ.bob
        pg.sprite.Sprite.__init__(self, self.groups)

        self.environ = environ
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
        self.stopBob =False

    def Collison(self):
        bobCollide=pg.sprite.spritecollide(self, self.environ.walls,False)
        if bobCollide:
            self.stopBob=True
            #print("bobCollide {} , stopBob {}, Bobpos {}".format(bobCollide,self.stopBob,(self.x,self.y)))


    def move(self, direction):
        if direction =="North":
            self.y -=1
            self.rect.y= self.y * TILESIZE

        elif direction =="South":
            self.y +=1
            self.rect.y = self.y * TILESIZE
        elif direction == "East":
            self.x += 1
            self.rect.x = self.x * TILESIZE
        else :
            self.x -= 1
            self.rect.x = self.x * TILESIZE

        #print("direction {}, x {}, y {}".format(direction,self.x,self.y))
        return (self.x,self.y)

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

        #self._layer = self.rect.bottom

class EndPoint(pg.sprite.Sprite):

    def __init__(self, environ, x, y, fittest):
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


        #self._layer = self.rect.bottom

class GA:
    def __init__(self, PopSize=1000, CrossOverRate=0.7, MutationRate=0.001, ChromoLength=70, GeneLength=2):
        self.ListGenomes = []
        self.PopSize = PopSize
        self.CrossOverRate = CrossOverRate
        self.MutationRate = MutationRate
        self.ChromoLength = ChromoLength
        self.GeneLength = GeneLength

        self.FittestGenome = 0
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0
        self.Busy =True

        self.map=Map()

    def CreateStartPopulation(self):
        for genome in range(0, self.PopSize):
            Genomes = {'vecBits': [randint(0, 1) for bit in range(0, self.ChromoLength)],
                       'Fitness': 0,
                       "EndPoint":(0,0),
                       "StartPoint":(0,0),
                       "Dist": 0
                       }
            self.ListGenomes.append(Genomes)

        self.FittestGenomeEver= {'vecBits': [0 for bit in range(0, self.ChromoLength)],
                                 'Fitness': 0,
                                 "EndPoint": (0, 0),
                                  "StartPoint":(0,0),
                                  "Dist": 0
                                }
        self.FittestGenome = self.FittestGenomeEver.copy()
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0
        self.Generation = 0


    def Mutate(self,vecBits):
        mutation=vecBits
        for index, bit in enumerate(vecBits):
            if   random() < self.MutationRate:
                if bit ==1:
                    mutation[index]=0
                else:
                    mutation[index] = 1

        return mutation

    def CrossOver(self, mum, dad):
        baby1 = mum.copy()
        baby2 = dad.copy()

        #just return parents as offspring dependent on the rate
	    #or if parents are the same
        if random() > self.CrossOverRate or mum['vecBits']==dad['vecBits']:
            baby1=mum.copy()
            baby2=dad.copy()
            return baby1, baby2

        #determine a crossover point
        crossoverPoint=randint(0, self.ChromoLength)

        baby1['vecBits']=(mum['vecBits'][0:crossoverPoint]).copy()+(dad['vecBits'][crossoverPoint:self.ChromoLength]).copy()
        baby2['vecBits']=(dad['vecBits'][0:crossoverPoint]).copy()+(mum['vecBits'][crossoverPoint:self.ChromoLength]).copy()

        baby1['Fitness'] = 0
        baby2['Fitness'] = 0

        return baby1, baby2

    def RouletteWheelSelection(self):
        slice=random()*self.TotalFitnessScore
        #print("slice--{}--{}".format(slice,self.TotalFitnessScore))
        total =0.0
        #selectedGenome=self.ListGenomes[0].copy()

        for genome in self.ListGenomes:
            total +=genome['Fitness']

            if total > slice:
                selectedGenome = genome.copy()
                break

        return selectedGenome

    def WeightedRouletteWheelSelection(self):
        #Find genome slice
        RouletteWheel=[]
        for genome in self.ListGenomes:
            slice= math.ceil((genome['Fitness']*100)/self.TotalFitnessScore)
            #print("total Fitness {} Fitness {} slice ={}".format(self.TotalFitnessScore,genome['Fitness'],slice))

            for size in range(0,slice):
                RouletteWheel.append(genome.copy())
        #print("------RouletteWheel{}".format(len(RouletteWheel)))
        selectedGenome = rd.choice(RouletteWheel)
        return selectedGenome

    def UpdateFitnessScores(self):
        self.BestFitnessScore = 0
        self.TotalFitnessScore = 0

        for genome in self.ListGenomes:
            #decode each genomes chromosome into a vector of directions
            directions=self.Decode(genome['vecBits'])

            #get it's fitness score
            genome['Fitness'],genome['EndPoint'],genome['StartPoint'],genome['Dist']=self.map.TestRoute(directions,True)
            #print("genome {}".format(genome))
            self.TotalFitnessScore +=genome['Fitness']

            if genome['Fitness'] > self.BestFitnessScore:
                self.BestFitnessScore=genome['Fitness']
                self.FittestGenome   =genome.copy()
                #print("BestFitnessScore {} genome {}  FittestEver {}".format(self.BestFitnessScore,genome['Fitness'],self.FittestGenomeEver['Fitness']))

                if genome['Fitness'] ==1:
                    self.Busy=False

            if self.FittestGenome['Fitness'] > self.FittestGenomeEver['Fitness']:
                self.FittestGenomeEver=self.FittestGenome.copy()


    def Decode(self,genome):
        i=-1
        gene=[]
        direction=[]

        for chromo in genome:
            i += 1
            gene.append(chromo)

            if i==1:
                direction.append(self.BinToInt(gene))
                i=-1
                gene=[]
        return direction


    def BinToInt (self, bin):
        val=0
        multiplier =1
       # print("{} bin".format(bin))
        for bit in reversed(bin):
            val += bit*multiplier
            multiplier *=2

        return val

    def Epoch(self):
        #print("BestBU{}".format(test.FittestGenome))
        self.UpdateFitnessScores()
        #print("BestAU{}".format(test.FittestGenome))
        NewBabies =0
        ListBabyGenomes=[]

        while NewBabies < self.PopSize:
            mum=self.WeightedRouletteWheelSelection()
            dad = self.WeightedRouletteWheelSelection()

            baby1,baby2=self.CrossOver(mum,dad)

            baby1['vecBits']=self.Mutate(baby1['vecBits'].copy())
            baby2['vecBits'] = self.Mutate(baby2['vecBits'].copy())

            ListBabyGenomes.append(baby1)
            ListBabyGenomes.append(baby2)
            NewBabies +=2

        self.ListGenomes= ListBabyGenomes.copy()
        self.ListGenomes.append(self.FittestGenome.copy())
        #print("Adding{} ---{}".format(self.ListGenomes[-1],self.FittestGenome.copy()))
        self.Generation +=1



if __name__ == '__main__':
    test = GA(PopSize=1000,ChromoLength=64)
    test.CreateStartPopulation()
    test.map.drawMaze()

    while test.Generation <100:
        test.Epoch()
        print("Best {}".format(test.FittestGenome))
        #test.map.TestRoute(test.FittestGenome['vecBits'], False)

        print("Ever {}".format(test.FittestGenomeEver))
        #test.map.TestRoute(test.FittestGenomeEver['vecBits'], False)
        print("-------------------------")
        test.map.EndPointdraw(test.FittestGenome['EndPoint'],False)
        test.map.EndPointdraw(test.FittestGenomeEver['EndPoint'], True)

        if test.FittestGenome['Fitness'] < test.FittestGenomeEver['Fitness']:
            break

        test.map.infoBoard("Generation {} best fitness score {:.4f}={:.4f},average Fitness {:.4f}-".format(test.Generation,test.BestFitnessScore,test.FittestGenome['Fitness'],test.TotalFitnessScore/(test.PopSize+1))) #test.FittestGenome['vecBits'],
        test.map.reset()
    print("Fin")
    #test.map.infoBoard("End")
    #test.map.TestRoute(test.FittestGenome['vecBits'],True)

    while True:
        FPS = 1
        directions = test.Decode(test.FittestGenomeEver['vecBits'])
        test.map.TestRoute(directions,False)
        test.map.reset()











