from numpy.lib.index_tricks import r_
import taichi as ti
import numpy as np
import math
import matplotlib.pyplot as plt
# from taichi.type.annotations import ext_arr


# ---------------------------------..-.....++.------------------------------------------ #
#                               paramters                              #
# ---------------------------------------------------------------------------- #
# global constant
numPar = 1000
density0 = 1.0
kernelRadius = 1.0
particleRadius = kernelRadius/4.0
mass = 1.0  # tweak
boundX = 100.0  # whole region size
boundY = 100.0
waterBoundX = 0.2 * boundX
waterBoundY = 0.2 * boundY
waterPosX = 0.3 * boundX
waterPosY = 0.2 * boundY
restiCoeff = 0.99
fricCoeff = 1.0  # not physical frictional coefficient
EosCoeff = 50.0  # coefficient of equation of state, copied from splishsplash
EosExponent = 7.0  # Gamma of equation of state
viscosity_mu = 1.0  # dynamic viscosity coefficient, the mu
# timeStepSize = 1e-4  # time step size
cellSize = 4.0  # not real cell, just for hashing
numCellX = ti.ceil(boundX/cellSize)   # number of cells in x direction
numCellY = ti.ceil(boundY/cellSize)   # number of cells in y direction
numCell = (numCellX)*(numCellY)
# number of cells, additional layer is to prevent out of bound

# ---------------------------------------------------------------------------- #
#                                end parameters                                #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                             BEGIN TAICHI PROGRAM                             #
# ---------------------------------------------------------------------------- #
DEBUG = False  # print debug info
PROFILE = False

if DEBUG == True:
    ti.init(arch=ti.cpu, debug=True, excepthook=True,
            cpu_max_num_threads=1, advanced_optimization=False)
elif PROFILE == True:
    ti.init(arch=ti.gpu, kernel_profiler=True)
else:
    ti.init(arch=ti.gpu)

@ti.func
def debugPrint(word,flag=True):
    if flag==True and DEBUG==True:
        print(word)

# ---------------------------------------------------------------------------- #
#                               tweak parameters                               #
# ---------------------------------------------------------------------------- #

paused = ti.field(ti.i32, shape=())
stepCount = ti.field(int, shape=())
paused[None] = False

timeStepSize = ti.field(ti.f32, shape=())
timeStepSize[None] = 1e-3

gravity = ti.Vector.field(2, ti.f32, shape=())
gravity[None] = ti.Vector([0, -100])
# ---------------------------------------------------------------------------- #
#                                physical fields                               #
# ---------------------------------------------------------------------------- #
# global field
position = ti.Vector.field(2, float, shape=numPar)
velocity = ti.Vector.field(2, float, shape=numPar)
density = ti.field(float, shape=numPar)
pressure = ti.field(float, shape=numPar)
acceleration = ti.Vector.field(2, float, shape=numPar)
pressureGradientForce = ti.Vector.field(2, float, shape=numPar)
viscosityForce = ti.Vector.field(2, float, shape=numPar)

#PCISPH
accNonP = ti.Vector.field(2, float, shape=numPar)
positionStar = ti.Vector.field(2, float, shape=numPar)
velocityStar = ti.Vector.field(2, float, shape=numPar)
densityErr = ti.field(float, shape=numPar)
delta = ti.field(float,())
avgDensityErr = ti.field(float,())

den1 = np.zeros((10), dtype = 'f4')  
pos1 = np.zeros((10), dtype = [('x', 'f4'), ('y', 'f4')])  
vel1 = np.zeros((10), dtype = [('x', 'f4'), ('y', 'f4')])  
acc1 = np.zeros((10), dtype = [('x', 'f4'), ('y', 'f4')])  
denErr1 = np.zeros((10), dtype = 'f4')  


# ---------------------------------------------------------------------------- #
#                            neighbor search variables                         #
# ---------------------------------------------------------------------------- #
# neighbor search related
maxNumNeighbors = 1000  # max len for neiList and cell2Par
maxNumParInCell = 1000

# sparse data structure: 
# place neighbor list
numNeighbor = ti.field(int)
neighbor = ti.field(int)
neighborNode = ti.root.bitmasked(ti.i, numPar)
neighborNode.place(numNeighbor)
neighborNode.bitmasked(ti.j, maxNumNeighbors).place(neighbor)

# place cell2Par
numParInCell = ti.field(int)
cell2Par = ti.field(int)
cell2ParNode = ti.root.bitmasked(ti.i, numCell)
cell2ParNode.place(numParInCell)
cell2ParNode.bitmasked(ti.j, maxNumParInCell).place(cell2Par)
# ---------------------------------------------------------------------------- #
#                         end neighbor search variables                        #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                          kernel Func and dirivatives                         #
# ---------------------------------------------------------------------------- #
# @ti.func
def kernelFunc(r):
    # poly6 kernel, r is the distance(r>0)
    res = 0.0
    h = kernelRadius
    if r < kernelRadius:
        x = (h * h - r * r) / (h * h * h)
        res = 315.0 / 64.0 / math.pi * x * x * x
    return res


# @ti.func
def firstDW(r):
    # first derivative of spiky kernel, r is the distance(r>0)
    res = 0.0
    h = kernelRadius
    if r < h:
        x = 1.0 - r / h
        res = -30.0 / (math.pi * h**3) * x * x
    return res


# @ti.func
def secondDW(r):
    # second derivative of kernel W
    # r must be non-negative
    h = kernelRadius
    res = 0.0
    if r < kernelRadius:
        x = 1.0 - r / h
        res = 60.0 / (math.pi * h**4) * x
    return res
# ---------------------------------------------------------------------------- #
#                        end kernel Func and dirivatives                       #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                neighbor search                               #
# ---------------------------------------------------------------------------- #


@ti.kernel
def neighborSearch():
    searchRadius = kernelRadius * 1.1
    # the search radius is not neccearily equals to the kernel radius
    # sometimes it> kernel radius, e.g. 1.1*kernelRadius

    # update hash table: cell2Par
    for par in range(numPar):
        cell = getCell(position[par])  # get cell ID from position

        k = ti.atomic_add(numParInCell[cell], 1)  # add the numParInCell by one
        # usage of ti.atomic_add： numParInCell[cell] add by one,
        #  old value of numParInCell[cell] is stored in k

        cell2Par[cell, k] = par

    # begin building the neighbor list
    for i in range(numPar):

        cell = getCell(position[i])
        # get cell ID from position

        kk = 0  # kk is the kkth neighbor in the neighbor list

        offs = ti.Vector([0, 1, -1,
                        -numCellX, -numCellX-1, -numCellX+1,
                        numCellX,  numCellX-1, numCellX+1])
        neiCellList = cell + offs
        # the 9 neighbor cells

        for ii in ti.static(range(9)):
            cellToCheck = neiCellList[ii]
            # which cell to check

            if IsInBound(cellToCheck):
                # prevent the cell ID out of bounds, which will happen in the boundary cells
                # after adding the offset. If out-of-bounds, do not check
                for k in range(numParInCell[cellToCheck]):
                    # kth particle in this cell

                    j = cell2Par[cellToCheck, k]
                    # j is another particle in this cell

                    if kk < maxNumNeighbors and j != i and \
                            (position[i] - position[j]).norm() < searchRadius:

                        # j is the kkth neighbor of particle i
                        neighbor[i, kk] = j

                        kk += 1
        numNeighbor[i] = kk


# the helper function for neighbor search
# check whether the cell ID out of bounds
@ti.func
def IsInBound(c):
    return c >= 0 and c < numCell


# the helper function for neighbor search
@ti.func
def getCell(pos):
    # 0.5 is for correct boundary cell ID, see doc for detail
    cellID = int(pos.x/cellSize - 0.5) + \
        int(pos.y/cellSize - 0.5)*numCellX

    # cellID=(position[i] / cellSize).cast(int)
    return cellID
# ---------------------------------------------------------------------------- #
#                              end neighbor search                             #
# ---------------------------------------------------------------------------- #
@ti.kernel
def computeDensity():
    eps = 0.1  # to prevent the density be zero, because it is denominator
    for i in density:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            r = (position[i]-position[j]).norm()
            density[i] += mass * kernelFunc(r)

        # to prevent the density be zero, because it is denominator
        if density[i] < eps:
            density[i] = eps


@ti.kernel
def computeViscosityForce():
    for i in viscosityForce:
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            r = (position[j]-position[i]).norm()

            viscosityForce[i] += mass * mass * viscosity_mu * (
                (velocity[j] - velocity[i]) / density[j]
            ) * secondDW(r)


# ---------------------------------------------------------------------------- #
#                               PciPressureSolver                              #
# ---------------------------------------------------------------------------- #
# ----------------------- prerequst for pciPressureSolver ---------------------- #

#计算不考虑压力的位置速度，相当于给x* v*初始化
# @ti.kernel
# def advanceTimeNonP():
#     for i in range(numPar):
#         velocityStar[i] = velocity[i] + accNonP[i] * timeStepSize[None]
#         positionStar[i] = position[i] + velocityStar[i] * timeStepSize[None]
        
@ti.kernel
def computeAccNonP():
    for i in accNonP:
        accNonP[i] = gravity[None]  \
            + viscosityForce[i] / mass 

        

#根据上一步迭代得到的压力梯度力，计算v*和x*, 算法步骤9-11
@ti.kernel
def predictVelocityPosition():
    for i in range(numPar):
        velocityStar[i] = velocity[i] + (accNonP[i] + pressureGradientForce[i]/mass)  * timeStepSize[None]
        positionStar[i] = position[i] + velocityStar[i] * timeStepSize[None]


# #计算rho*，完全与常规SPH一致,除了位置换成positionStar, density换成densityStar
# @ti.kernel
# def computeDensityStar():
#     eps = 0.1  # to prevent the density be zero, because it is denominator
#     for i in densityStar:
#         densityStar[i] =0.0

#     for i in densityStar:
#         for k in range(numNeighbor[i]):
#             j = neighbor[i, k]
            
#             r = (positionStar[i]-positionStar[j]).norm() #这里位置换成了x*
#             densityStar[i] += mass * kernelFunc(r)

#         # to prevent the densityStar be zero, because it is denominator
#         if densityStar[i] < eps:
#             densityStar[i] = eps



# # 计算密度误差和平均密度误差
# @ti.kernel
# def computeDensityErr()->ti.f32:
#     avgDensityErr=0.0
#     for i in densityStar:
#         densityErr[i]=0.0
#     for i in densityStar:
#         #计算密度误差和平均密度误差
#         densityErr[i] = densityStar[i] - density0
#         avgDensityErr += densityErr[i]
#         # print("densityErr[",i,"]",densityErr[i])
#         # print("avgDensityErr",avgDensityErr)
#     avgDensityErr /= numPar
#     return avgDensityErr



#计算PCI系数
def computeDelta():
    h=kernelRadius
    sumGradW2 =0.0
    sumGradW = np.array([0.0, 0.0])
    xi = np.array([0, 0])
    xj = np.array([-h,-h])
    diam= 2 * particleRadius
    r=np.zeros(2)
    while(xj[0] <= h):
        while(xj[1] <= h):
            r = xi-xj
            r_mod =  np.linalg.norm(r)
            if 1e-6 <= r_mod <= h:
                gradW = firstDW(r_mod) * r / r_mod
                sumGradW += gradW
                sumGradW2 += gradW.dot(gradW)
            xj[1] += diam
        xj[0] += diam
        xj[1] = -h
    beta =  2.0 * (mass / density0 * timeStepSize[None] ) **2 
    delta[None]= 1.0 /  ( beta* (sumGradW.dot(sumGradW) + sumGradW2 ))
    # print("delta[None]",delta[None])


#FIXME:
@ti.kernel
def computePressure():
    avgDensityErr[None] = 0.0
    for i in range(numPar):
        sumW = 0.0 
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]
            r = (positionStar[i]-positionStar[j]).norm() #这里位置换成了x*
            sumW +=  kernelFunc(r)
            # if i== 94:
            #     print("No.",k,":",j)
            #     print("r:",r)
            #     print("kernel(r)",kernelFunc(r))
            #     print()
        # sumW += kernelFunc(0)

        tempDen = mass * sumW
        tempDenErr = tempDen - density0
        pTilde =  delta[None] * tempDenErr #p~

        if pTilde < 0.0:
            pTilde = 0.0
            tempDenErr = 0.0
        # if i==94:
        #     print(i,"tempDen",tempDen)

        pressure[i] += pTilde
        density[i] += tempDen
        densityErr[i] += tempDenErr

        # if i==0: print("################\n\n")
        # if i==33:
        #     print(i,"density",density[i])
        avgDensityErr[None] += densityErr[i]

    # np.savetxt("densityErr.csv",densityErr,delimiter=',')
    avgDensityErr[None] /= numPar
    


#TODO:
@ti.kernel
def computePressureGradientForce():
    for i in pressureGradientForce:
        sumFp=ti.Vector([0.0,0.0])
        for k in range(numNeighbor[i]):
            j = neighbor[i, k]

            # grad Wij
            dir = (positionStar[i]-positionStar[j]).normalized()
            r = (positionStar[i]-positionStar[j]).norm()
            gradW = firstDW(r) * dir

            sumFp -= mass * mass * (
                pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2
            ) * gradW
        pressureGradientForce[i] = sumFp


# # 把速度位置和密度复制到v* x* rho*
# @ti.kernel
# def copyVelPosDenStar():
#     for i in range(numPar):
#         velocityStar[i] = velocity[i]
#         positionStar[i] = position[i]
#         densityStar[i]  = density[i]


@ti.kernel
def computeAcceleration():
    for i in acceleration:
        acceleration[i] += gravity[None]  \
            + viscosityForce[i] / mass  \
            + pressureGradientForce[i] / mass

 # ---------------------------------------------------------------------------- #
 #                             PciPressureSolver                            #
 # ---------------------------------------------------------------------------- #
#TODO:
def PciPressureSolver():
    # print("\n\n PCI begin")
    # TESTPrintParticleInfo(0)

    computeAccNonP()
    # print("accNonP[0]",accNonP[0])

    pressure.fill(0)
    pressureGradientForce.fill(0)

    avgDensityErr[None]=1.0 #压力误差
    maxIter=10
    iter=0


    # print("density[33]",density[33])

    while ti.abs(avgDensityErr[None]/density0)>1e-2 and iter <maxIter:
    # while iter<10:
    #     print()
    #     print()
    #     print("iter=",iter)

        predictVelocityPosition() #计算预测速度位置

        boundaryCollisionStar() #边界处理，与常规SPH完全一致

        computePressure() #计算p和densityErr

        computePressureGradientForce() #计算压力梯度力

        # print("\nafter Fp")
        # TESTPrintParticleInfo1(0)
        # print("avgDensityErr=",avgDensityErr[None])
        # print("densityErr[0]=",densityErr[80])
        iter+=1
    # print(iter)
        

 # ---------------------------------------------------------------------------- #
 #                             end PciPressureSolver                            #
 # ---------------------------------------------------------------------------- #

@ti.kernel
def boundaryCollisionStar():
    eps = 0.1
    for i in range(numPar):
        # left
        if positionStar[i].x < 0.0:
            positionStar[i].x = 0.0
            velocityStar[i].x *= -restiCoeff
            velocityStar[i].y *= fricCoeff #not physical but work
            # print("BC")

        # right
        elif positionStar[i].x >= boundX - eps:
            positionStar[i].x = boundX - eps
            velocityStar[i].x *= -restiCoeff
            velocityStar[i].y *= fricCoeff #not physical but work
            # print("BC")

        # top
        elif positionStar[i].y >= boundY - eps:
            positionStar[i].y = boundY - eps
            velocityStar[i].y *= -restiCoeff
            velocityStar[i].x *= fricCoeff #not physical but work
            # print("BC")

        # bottom
        elif positionStar[i].y < 0.0:
            positionStar[i].y = 0.0
            velocityStar[i].y *= -restiCoeff
            velocityStar[i].x *= fricCoeff #not physical but work
            # print("BC")


@ti.kernel
def boundaryCollision():
    eps = 0.1
    for i in range(numPar):
        # left
        if position[i].x < 0.0:
            position[i].x = 0.0
            velocity[i].x *= -restiCoeff
            velocity[i].y *= fricCoeff #not physical but work

        # right
        elif position[i].x >= boundX - eps:
            position[i].x = boundX - eps
            velocity[i].x *= -restiCoeff
            velocity[i].y *= fricCoeff #not physical but work

        # top
        elif position[i].y >= boundY - eps:
            position[i].y = boundY - eps
            velocity[i].y *= -restiCoeff
            velocity[i].x *= fricCoeff #not physical but work

        # bottom
        elif position[i].y < 0.0:
            position[i].y = 0.0
            velocity[i].y *= -restiCoeff
            velocity[i].x *= fricCoeff #not physical but work


@ti.kernel
def advanceTime():
    for i in range(numPar):
        velocity[i] += acceleration[i] * timeStepSize[None]
        position[i] += velocity[i] * timeStepSize[None]


@ti.kernel
def initialization():
    for i in range(numPar):
        # aligned init:
        r = waterBoundX/waterBoundY
        a = waterBoundX*waterBoundY/numPar
        dx = ti.sqrt(a*r)
        dy = ti.sqrt(a/r)

        perRow = (waterBoundX/dx)

        position[i] = [
            i % perRow * dx + waterPosX,
            i // perRow * dy + waterPosY,
        ]


def draw(gui):

    # normalize position data (in (0.0,1.0)) for drawing
    pos = position.to_numpy()
    pos[:, 0] *= 1.0 / boundX
    pos[:, 1] *= 1.0 / boundY

    # draw the particles
    gui.circles(pos,
                radius=2.0,
                )


def clear():
    # clear the density
    density.fill(0.0)

    # clear the forces and acceleration
    acceleration.fill(0.0)
    pressureGradientForce.fill(0.0)
    viscosityForce.fill(0.0)

    # clear the neighbor list and cell2Par
    numParInCell.fill(0)
    numNeighbor.fill(0)
    neighbor.fill(-1)  # because the cell ID begin with 0, default should be -1
    cell2Par.fill(0)


# ---------------------------------------------------------------------------- #
#                                   test func                                  #
# ---------------------------------------------------------------------------- #
def TESTKernel():
    h = kernelRadius

    r = np.zeros(1000)
    y = np.zeros(1000)
    y1 = np.zeros(1000)
    y2 = np.zeros(1000)
    for i in range(1000):
        r[i] = i * h/1000
        y[i] = kernelFunc(r[i])
        y1[i] = firstDW(r[i])
        y2[i] = secondDW(r[i])
    plt.plot(r, y, 'r')
    plt.plot(r, y1, 'g')
    plt.plot(r, y2, 'b')
    plt.show()
    # np.savetxt("y.csv", y, delimiter=',')
    # np.savetxt("y1.csv", y1, delimiter=',')
    # np.savetxt("y2.csv", y2, delimiter=',')
    np.savetxt("kernelFunc.csv", [y, y1, y2], delimiter=',')


def TESTTwoPar():
    # init with only two particles
    # set numPar=2
    # cancel gravity
    gui = ti.GUI("SPHDamBreak",
                background_color=0x112F41,
                res=(1000, 1000)
                )

    distX = 5*kernelRadius/10.0
    distY = 0.0  # 5*kernelRadius/10.0

    position[0] = [
        boundX / 2.0,
        boundY / 2.0,
    ]
    position[1] = [
        boundX/2.0 + distX,
        boundY/2.0 + distY,
    ]

    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(1):
            step()
        draw(gui)


def TESTPrintNeighbor(i:ti.i32):
    print("########neighbor of ", i, '#########')
    print('numNeighbor=', numNeighbor[i])
    print('current position={}'.format(position[i]))

    for k in range(numNeighbor[i]):
        j = neighbor[i, k]
        print('No.{} neighbor:{}'.format(k, j))
        print('position={}'.format(position[j]))
        # rr=(position[i]-position[j])
        # r_mod=rr.to_numpy()

        # print("r_mod=",r_mod)
        # print("kernelFunc=",kernelFunc(r_mod))
        print()
    print("########END neighbor of ", i, '#########')


def TESTPrintParticleInfo1(i):
    print('###########print particle info {}###########'.format(i))
    print('velocityStar[{}]={}'.format(i, velocityStar[i]))
    print('positionStar[{}]={}'.format(i, positionStar[i]))
    print('pressureGradientForce[{}]={}'.format(i, pressureGradientForce[i]))

    # print('*********END print particle info {}*********'.format(i))

def TESTPrintParticleInfo(i):
    print('###########print particle info {}###########'.format(i))
    print('density[{}]={}'.format(i, density[i]))
    print('pressure[{}]={}'.format(i, pressure[i]))
    print('velocity[{}]={}'.format(i, velocity[i]))
    print('position[{}]={}'.format(i, position[i]))
    print('*********END print particle info {}*********'.format(i))

def TESTPrintParticleInfoStar(i):
    print('###########print particle info star {}###########'.format(i))
    # print('densityStar[{}]={}'.format(i, densityStar[i]))
    print('pressure[{}]={}'.format(i, pressure[i]))
    print('pressureGradientForce[{}]={}'.format(i, pressureGradientForce[i]))
    print('velocityStar[{}]={}'.format(i, velocityStar[i]))
    print('positionStar[{}]={}'.format(i, positionStar[i]))
    print('densityErr[{}]={}'.format(i, densityErr[i]))

    print('*********END print particle info star {}*********'.format(i))

#test wheter the sparse sturcture works
@ti.kernel
def TESTPrintSparseStructure():
    # print('numCell = ', numCell)
    usedCell=0
    for c in range(numCell):
        #for k in range(maxNumParInCell):
            if ti.is_active(cell2ParNode, [c,0]):
                usedCell+=1
                print("cell",c, " is used")
    print('usedCell = ', usedCell)
    

def TestRunOnce():
    step()


# ---------------------------------------------------------------------------- #
#                                 end test func                                #
# ---------------------------------------------------------------------------- #


def run():
    # constantly run
    gui = ti.GUI("SPHDamBreak",
                background_color=0x112F41,
                res=(1000, 1000)
                )

    paused[None] = False

    while ti.GUI.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()

            # press space to pause
            elif e.key == gui.SPACE:
                paused[None] = not paused[None]

            # press s to step once
            elif e.key == 's':
                for s in range(1):
                    step()

        if not paused[None]:
            for s in range(100):
                step()

        draw(gui)

def TEST2numpy():
    pos1=position.to_numpy()
    vel1=velocity.to_numpy()
    acc1=acceleration.to_numpy()
    den1=density.to_numpy()


def step():
    stepCount[None] += 1
    clear()
    neighborSearch()
    computeDensity()
    computeViscosityForce()

    # TEST2numpy()

    PciPressureSolver()#计算之后得到压力梯度力

    computeAcceleration()
    advanceTime()
    boundaryCollision()

    # for i in range(33,50,1):
    #     TESTPrintNeighbor(i)
    #     print('density[{}]={}'.format(i, density[i]))
def run():
    # constantly run
    gui = ti.GUI("PCISPH",
                 background_color=0x112F41,
                 res=(500, 500)
                 )

    frame=0

    paused[None] = False

    while True:
        for e in gui.get_events(ti.GUI.PRESS):

            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()

            # press space to pause
            if e.key == gui.SPACE:
                paused[None] = not paused[None]

            # press s to step once
            elif e.key == 's':
                for s in range(1):
                    computeDelta()
                    step()

        if not paused[None]:
            for s in range(100):
                computeDelta()
                step()

        draw(gui)

        # gui.show(f'data/temp/{frame:06d}.png')
        # frame+=1
        gui.show()


if __name__ == '__main__':
    initialization()
    run()

    # step()
    pass