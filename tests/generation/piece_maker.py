import numpy
import os
import sys
from PIL import Image, ImageDraw

t_FEMALE = 1
t_MALE = 2
t_LINE = 3

def computeBezierPoint(points, t):
    tSquared = t * t 
    tCubed = tSquared * t 

    cx = 3.0 * (points[1][0] - points[0][0])
    bx = 3.0 * (points[2][0] - points[1][0]) - cx
    ax = points[3][0] - points[0][0] - cx - bx

    cy = 3.0 * (points[1][1] - points[0][1])
    by = 3.0 * (points[2][1] - points[1][1]) - cy
    ay = points[3][1] - points[0][1] - cy - by

    x = (ax * tCubed) + (bx * tSquared) + (cx * t) + points[0][0]
    y = (ay * tCubed) + (by * tSquared) + (cy * t) + points[0][1]

    return (x,y)

def computerBezier(points,num):
    dt = 1.0/num
    curvePoints = []

    for i  in range(0,num) :
        p = computeBezierPoint(points,dt * i )
        curvePoints.append(p)

    return curvePoints 


def polygonCropImage(im,polygon,name):
    imArray = numpy.asarray(im)

    # create mask
    # polygon = [(0,0),(0,200),(200,200)]
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)
    
    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')
    
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    newIm.save(name)
    
#we assume the (0,0) is at the center of the rectangle
class PieceOutLine():
    def __init__(self,width,height,ar,cr):
        self.w = width 
        self.h = height 
        self.arcRatio = ar
        self.connectRatio = cr
        self.pointNum = 300
         
    def genRightFemaleArc(self,istop):
        halfW = self.w * 0.5
        halfH = self.h * 0.5
        arcW = self.w * self.arcRatio 
        connectW = self.h * self.connectRatio

        top = (halfW,-halfH) 
        bottom = (halfW + arcW, -connectW * 0.5)
        dw = bottom[0] - top[0]
        dh = bottom[1] - top[1]

        points = [ top,(top[0] +  dw/3 + 8,top[1] + dh/3),(top[0] + 2*dw/3 + 8,top[1] + dh * 2/3),bottom ]

        curvPoints = [] 
        if not istop:
            curvPoints = computerBezier(points, self.pointNum )
        else:
            left = []
            for p in reversed(points):
                left.append((p[0],-p[1]))
            curvPoints = computerBezier(left, self.pointNum )

        return curvPoints

    def genRightFemaleConnect(self,left):
        halfW = self.w * 0.5
        halfH = self.h * 0.5
        arcW = self.w * self.arcRatio 
        connectW = self.h * self.connectRatio 

        startX = halfW + arcW
        startY = -connectW * 0.5

        endX = startX - connectW
        endY = 0
        points = [ 
                (startX,startY),
                (endX + (startX - endX)*3/5, -startY*2/3),
                (endX,2 * startY),
                (endX,endY)
                ]

        curvPoints = [] 
        if not left:
            curvPoints = computerBezier(points, self.pointNum )
        else:
            left = []
            for p in reversed(points):
                left.append((p[0],-p[1]))
            curvPoints = computerBezier(left, self.pointNum )

        return curvPoints

    def genBottomFemaleArc(self,isLeft):
        halfW = self.w * 0.5
        halfH = self.h * 0.5
        arcH = self.h * self.arcRatio 
        connectW = self.w * self.connectRatio 
        
        right = (halfW,halfH)
        left =  (connectW * 0.5,halfH + arcH)
        dw = right[0] - left[0]
        dh = left[1] - right[1]

        points = [ right,(right[0] - dw/3,right[1] + dh/3 + 8),(right[0] - 2*dw/3,right[1] + dh * 2/3 + 8),left]
        

        curvPoints = [] 
        if not isLeft:
            curvPoints = computerBezier(points, self.pointNum )
        else:
            left = []
            for p in reversed(points):
                left.append((-p[0],p[1]))
            curvPoints = computerBezier(left, self.pointNum )

        return curvPoints

    def genBottomFemaleConnect(self,left):
        halfW = self.w * 0.5
        halfH = self.h * 0.5
        arcH = self.h * self.arcRatio 
        connectW = self.w * self.connectRatio 

        startX = connectW * 0.5
        startY = halfH + arcH

        endX = 0 
        endY = startY - connectW
        points = [ 
                (startX,startY),
                (-startX*2/3 ,endY + (startY - endY)*3/5 ),
                (2*startX,endY),
                (endX,endY)
                ]

        curvPoints = [] 
        if not left:
            curvPoints = computerBezier(points, self.pointNum )
        else:
            left = []
            for p in reversed(points):
                left.append((-p[0],p[1]))
            curvPoints = computerBezier(left, self.pointNum )

        return curvPoints
    
    def genRightFemale(self):
        rightArc = self.genRightFemaleArc(False) 
        right = self.genRightFemaleConnect(False) 
        left = self.genRightFemaleConnect(True) 
        leftArc = self.genRightFemaleArc(True) 
        
        curvPoints = rightArc + right + left + leftArc
        return curvPoints

    def genRightMale(self):
        halfW = self.w * 0.5
        points = self.genRightFemale()
        curvPoints = [] 
        for p in points:
            curvPoints.append((p[0] + (halfW - p[0]) * 2, p[1]) )
        return curvPoints
    
    def genRightLine(self):
        return [ ( self.w *0.5, self.h *0.5 ) ]

    def genLeftMale(self):
        halfW = self.w * 0.5
        points = self.genRightFemale()
        curvPoints = [] 
        for p in points:
            curvPoints.append( (p[0] - halfW * 2, p[1]) )
        return reversed(curvPoints)

    def genLeftFemale(self):
        halfW = self.w * 0.5
        points = self.genLeftMale()
        curvPoints = [] 
        for p in points:
            curvPoints.append(((-p[0] - halfW)*2 + p[0],p[1]))
        return curvPoints

    def genLeftLine(self):
        return [ ( -self.w*0.5, -self.h*0.5 ) ] 

    def genBottomFemale(self):
        rightArc = self.genBottomFemaleArc(False) 
        right = self.genBottomFemaleConnect(False) 
        left = self.genBottomFemaleConnect(True) 
        leftArc = self.genBottomFemaleArc(True) 
        
        curvPoints = rightArc + right + left + leftArc
        return curvPoints

    def genBottomMale(self):
        halfH = self.h * 0.5
        points = self.genBottomFemale()
        curvPoints = [] 
        for p in points:
            curvPoints.append((p[0],(halfH - p[1])*2 + p[1] ))
        return curvPoints

    def genBottomLine(self):
        return [ ( -self.w*0.5, self.h*0.5 ) ] 

    def genTopMale(self):
        points = self.genBottomFemale()
        curvPoints = [] 
        for p in points:
            curvPoints.append((p[0],p[1] - self.h))

        return reversed(curvPoints)

    def genTopFemale(self):
        halfH = self.h * 0.5
        points = self.genTopMale()
        curvPoints = [] 
        for p in points:
            curvPoints.append((p[0],(-p[1] - halfH )*2 + p[1] ))
        return curvPoints

    def genTopLine(self):
        return [ ( self.w*0.5, -self.h*0.5 ) ] 
    
    def genOutLine(self,pieceBorders):
        curvPoints = [ ] 
        curvPoints.append((self.w * 0.5, self.h *0.5))
        func = [ 
                { 
                    t_FEMALE:   self.genBottomFemale,
                    t_MALE:     self.genBottomMale,
                    t_LINE:     self.genBottomLine,
                },

                { 
                    t_FEMALE:   self.genLeftFemale,
                    t_MALE:     self.genLeftMale,
                    t_LINE:     self.genLeftLine,
                },
                { 
                    t_FEMALE:   self.genTopFemale,
                    t_MALE:     self.genTopMale,
                    t_LINE:     self.genTopLine,
                },
                { 
                    t_FEMALE:   self.genRightFemale,
                    t_MALE:     self.genRightMale,
                    t_LINE:     self.genRightLine,
                },
        ]
        
        i =  0 
        for f in func:
            curvPoints += f[pieceBorders[i]]()
            i = i + 1 

        return curvPoints


class PieceInfo():
    def __init__(self,size,rowNum,colNum,ar,cr):
        self.w = size[0]/colNum
        self.h = size[1]/rowNum
        self.rowNum = rowNum 
        self.colNum = colNum 
        self.arcRatio = ar
        self.connectRatio = cr

    def getPieceInfo(self,row,col):
        arcW = self.w * self.arcRatio 
        arcH = self.h * self.arcRatio 
        connectW = self.w * self.connectRatio 
        connectH = self.h * self.connectRatio 
        borders = [] 

        t = t_MALE

        if (row + col)%2 == 0 :
            t = t_FEMALE

        if t == t_FEMALE:
            borders = [t_MALE,t_FEMALE,t_MALE,t_FEMALE] 
        else:
            borders = [t_FEMALE,t_MALE,t_FEMALE,t_MALE] 

        if col == 0:
            borders[1] = t_LINE

        if row == 0:
            borders[2] = t_LINE
        
        if (row + 1) == self.rowNum:
            borders[0] = t_LINE

        if (col + 1) == self.colNum:
            borders[3] = t_LINE

        topX = self.w * col 
        topY =  self.h *row 
        bottomX = self.w * (col + 1)
        bottomY = self.h *(row + 1)
        centerX = self.w * 0.5 
        centerY = self.h * 0.5

        #bottom
        if borders[0] == t_MALE:
            bottomY += (connectW - arcH)
        elif borders[0] == t_FEMALE:
            bottomY += arcH

        #left
        if borders[1] == t_MALE:
            topX -= (connectH - arcW)
            centerX += (connectH - arcW)
        elif borders[1] == t_FEMALE:
            topX -= arcW
            centerX += arcW
        #Top
        if borders[2] == t_MALE:
            topY -= (connectW - arcH)
            centerY += (connectW - arcH)
        elif borders[2] == t_FEMALE:
            topY -= arcH
            centerY += arcH

        #right
        if borders[3] == t_MALE:
            bottomX += (connectH - arcW)
        elif borders[3] == t_FEMALE:
            bottomX += arcW

        return (
                int(round(topX)),
                int(round(topY)),
                int(round(bottomX)),
                int(round(bottomY))
                ),(centerX,centerY),borders

def createPuzzlePieces(name,row,col,outPrefix):
    im = Image.open(name).convert("RGBA")
    #arcRatio = 0.0001
    #connectRatio = 0.2
    arcRatio = 0.09
    connectRatio = 0.3
    r = 3

    info = PieceInfo(im.size,row,col,arcRatio,connectRatio)

    outLine = PieceOutLine(im.size[0]/col,im.size[1]/row,arcRatio,connectRatio)
    
    w =im.size[0]/col
    h = im.size[1]/row

    draw = ImageDraw.Draw(im)
    outLinePoints = [] 
    json="{"
    first = True
    for i in range (0,row):
        for j in range (0,col):
            rect,center,borders = info.getPieceInfo(i,j)
            name = outPrefix + str(i) + "_" + str(j)
            if not first:
                json +=","
            first = False

            json +="\n    \"" + os.path.basename(name) + "\":[" + str(rect[0]) + "," + str(rect[1]) + "]" 
            region = im.crop(rect)
            curvPoints = outLine.genOutLine(borders) 

            cropPoints = [] 
            for p in curvPoints:
                cropPoints.append((p[0] + center[0],p[1] + center[1]))
            polygonCropImage(region,cropPoints,name + ".png")
            
            for p in curvPoints:
                outLinePoints.append((p[0] + j * w  + 0.5 * w ,p[1] + i * h + 0.5 * h))
    
    json +="\n}\n"
    dataFile = open(outPrefix + "data.json" ,"w")
    dataFile.write(json)
    dataFile.flush()
    dataFile.close()

    bgColor = (255,255,255,0)
    lineColor = (0,0,0,255)

    outLinedraw = ImageDraw.Draw(im)

    outLinedraw.rectangle( [ (0,0), (im.size[0],im.size[1]) ],fill = bgColor,outline = bgColor)

    for p in outLinePoints:
        px = p[0]
        py = p[1]
        outLinedraw.ellipse( (px - r, py - r, px + r, py + r ), fill=lineColor ,outline = lineColor)
    
    for x in range(0,im.size[0]):
        px = x
        py = 0.5 * r
        outLinedraw.ellipse( (px - r, py - r, px + r, py + r ), fill=lineColor,outline = lineColor)
        py = im.size[1] - 0.5 * r
        outLinedraw.ellipse( (px - r, py - r, px + r, py + r ), fill=lineColor,outline = lineColor)
    
    for y in range(0,im.size[1]):
        px = 0.5 * r
        py = y
        outLinedraw.ellipse( (px - r, py - r, px + r, py + r ), fill=lineColor,outline = lineColor)
        px = im.size[0] - 0.5 * r
        outLinedraw.ellipse( (px - r, py - r, px + r, py + r ), fill=lineColor,outline = lineColor)

    im.save(outPrefix + "outline.png")
    #im.show()




def main():
    if len (sys.argv) < 3:
        print(sys.argv[0] + " image row column")
        return

    outDir="_pieces/" + os.path.splitext(os.path.basename(sys.argv[1]))[0]

    if not os.path.exists(outDir) :
        os.makedirs(outDir)
    
    outPrefix =outDir + "/piece_"
    createPuzzlePieces(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),outPrefix)
    

if __name__ == "__main__":
    main()

