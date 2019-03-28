#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:38:24 2019

@author: mvb
"""
import preprocess as prep

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys, os
import pandas as pd

from pyqtgraph.dockarea import DockArea, Dock
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from scipy.ndimage.filters import gaussian_filter1d

class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, name):
        self.addChild(dict(name="%s" % (str(name)), type='bool', value=True, removable=True, renamable=True, expanded = False))


class Clarity(QtGui.QMainWindow):
    
    def __init__(self):
        super(Clarity, self).__init__()
        self.pathRootData = '/home/mvb/0_ETH/MasterThesis/Logs_GoKart/LogData/dynamics'
        testDays = self.getDirectories(self.pathRootData)
        testDays.sort()
        defaultDay = testDays[-1]
        self.pathTestDay = self.pathRootData + '/' + defaultDay
        LogNrs = self.getDirectories(self.pathTestDay)
        LogNrs.sort()
        defaultLogNr = LogNrs[21]
        self.pathLogNr = self.pathTestDay + '/' + defaultLogNr
        
        params = [
                {'name': 'Testing day', 'type': 'list', 'values': testDays, 'value': defaultDay},
                {'name': 'Log Nr.', 'type': 'list', 'values': LogNrs, 'value': defaultLogNr},
                {'name': 'Add to plot -->', 'type': 'action'},
                ScalableGroup(name='Data in Plot')
        ]
        self.p = Parameter.create(name='params', type='group', children=params)
        self.p.param('Testing day').sigValueChanged.connect(self.testDayChange)
        self.p.param('Log Nr.').sigValueChanged.connect(self.logNrChange)
        self.p.param('Add to plot -->').sigActivated.connect(self.addToPlot)
        self.p.param('Data in Plot').sigTreeStateChanged.connect(self.treeChange)
        
        self.dataList = QtGui.QListWidget()
        self.dataList.setSelectionMode(self.dataList.ExtendedSelection)
        self.plotfield = pg.PlotWidget()
        self.legend = pg.LegendItem()
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.p, showTop=False)
        
        self.dataList.itemSelectionChanged.connect(self.dataSelectionChanged)
        
        samplingRate = 10 #Hz
        self.timeStep = 1./samplingRate
        self.availableData = []
        self.plotDataList = []
        self.createUI()
        
        self.xrangefix = [0,0]
        self.yrangefix = [0,0]
        
    def createUI(self):
        dockArea = DockArea()
        self.setCentralWidget(dockArea)
        windowWidth = 1200
        windowHeight = 675
        self.setGeometry(100,100, windowWidth, windowHeight)
        self.setWindowTitle('Clarity')
        
        pg.setConfigOptions(antialias=True)
        
        d1 = Dock("Display Data", size=(windowWidth/3., windowHeight/3.))
        d2 = Dock("Plot 1", size=(windowWidth*2/3., windowHeight))
        d3 = Dock("Parameter Tree", size=(windowWidth/3., windowHeight*2./3))
        
        dockArea.addDock(d1,'left')
        dockArea.addDock(d2,'right')
        dockArea.addDock(d3,'bottom', d1)
        
        item = QtGui.QListWidgetItem(self.dataList)  
        item.setText('helloworld')
        
        d1.addWidget(self.dataList)
        
        d2.addWidget(self.plotfield)
        
        self.setListItems()
        d3.addWidget(self.tree)
        
        self.show()
        
    def filterChange(self, group, param):
        print('filter changed!')
        for param, value, data in param:
            if param.name() == 'sigma':
                for item in self.dataList.findItems(group.name(), QtCore.Qt.MatchExactly):
                    item.info[1] = data
                    item.info[2] = 10*data
            if param.name() == 'scale':
                for item in self.dataList.findItems(group.name(), QtCore.Qt.MatchExactly):
                    item.info[4] = data
        self.updatePlot()
        
    def testDayChange(self, param):
        self.plotfield.clear()
        self.p.param('Data in Plot').clearChildren()
        self.plottedData = []
        self.tempX = []
        self.tempY = []
        
        value = param.value()
        self.pathTestDay = self.pathRootData + '/' + value
        LogNrs = self.getDirectories(self.pathTestDay)
        LogNrs.sort()
        self.p.param('Log Nr.').remove()
        child = Parameter.create(name='Log Nr.', type='list', values = LogNrs, value=LogNrs[0])
        self.p.insertChild(1,child)
        self.p.param('Log Nr.').sigValueChanged.connect(self.logNrChange)
        self.pathLogNr = self.pathTestDay + '/' + LogNrs[0]

    def logNrChange(self, param):
        self.plotfield.clear()
        self.p.param('Data in Plot').clearChildren()
        self.availableData = []
        self.tempX = []
        self.tempY = []
        value = param.value()
        if value != None:
            self.pathLogNr = self.pathTestDay + '/' + value
        self.setListItems()
        
    def treeChange(self, group, param):
        print('Tree changed!')
        for param, value, data in param:
            print('  action:', value)
            if param.name() == 'Data in Plot':
                if value == 'childAdded':
                    for item in self.dataList.findItems(data[0].name(), QtCore.Qt.MatchExactly):
                        self.plotDataList.append(data[0].name())

                if value == 'childRemoved' and len(self.plotDataList) > 0:
                    for i in range(len(self.plotDataList)):
                        print(self.plotDataList, i)
                        if self.plotDataList[i] == data.name():
                            deleteElement = i
                    del self.plotDataList[deleteElement]
                self.updatePlot()
            elif value == 'value':
                for item in self.dataList.findItems(param.name(), QtCore.Qt.MatchExactly):
                    item.info[0] = data
                self.updatePlot()


    
    def dataSelectionChanged(self):
        print('dataselection changed')
        selected = self.dataList.selectedItems()
        if len(selected) > 2:
            self.dataList.blockSignals(True)
            try:
                for item in selected[1:-1]:
                    item.setSelected(False)
            finally:
                self.dataList.blockSignals(False)
        if len(selected) > 0:
            self.updatePlot()
    
    
    def addToPlot(self):
        sel = list([str(item.text()) for item in self.dataList.selectedItems()])
        dispName = 0
        if len(sel) == 1:
            dispName = sel[0]
        if len(sel) == 2:
            dispName = str(sel[0]) + ' / ' + str(sel[1])

        if (not(any([item.name() == dispName for item in self.p.param('Data in Plot').children()])) and dispName != 0):
            self.p.param('Data in Plot').addNew(dispName)
            for item in self.dataList.findItems(dispName, QtCore.Qt.MatchExactly):
                sigma = item.info[1]
                scale = item.info[4]
            OptionVars = [
                            {'name': 'sigma', 'type': 'float', 'value': sigma, 'step': 0.1},
                            {'name': 'scale', 'type': 'float', 'value': scale, 'step': 0.1},
                        ]
#            child = Parameter.create(name='Gaussian Filter', type='group', children = filterVars)
            self.p.param('Data in Plot').param(dispName).addChildren(OptionVars)
            self.p.param('Data in Plot').param(dispName).sigTreeStateChanged.connect(self.filterChange)
        elif dispName == 0:
            print('No Data Selected!')
        else:
            print(dispName + ' is already plotted!')
          
    def updateData(self, dataNames):
        availableDataList = [item[0] for item in self.availableData]
        for name in dataNames:
            if name in self.needsPreprocessing:
                for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
                    nameDependency = item.dependencies[0]
                for depend in nameDependency:
                    if depend not in dataNames:
                        index = dataNames.index(name)
                        dataNames = dataNames[:index] + [depend] + dataNames[index:]
        print('dataNames', dataNames)
        for name in dataNames:
            if name in self.needsPreprocessing:
                prep.preProcessing(self, name)
                availableDataList = [item[0] for item in self.availableData]
                for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
                    sigma = item.info[1]
                    width = item.info[2]
                index = availableDataList.index(name)
                yOld = self.availableData[index][2]
                trunc = (((width - 1)/2)-0.5)/sigma
                yNew = gaussian_filter1d(yOld,sigma,truncate=trunc)
                self.availableData[index][2] = yNew
            else:
                for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
                    sigma = item.info[1]
                    width = item.info[2]
                    xOld = item.data[0]
                    yOld = item.data[1]
                trunc = (((width - 1)/2)-0.5)/sigma
                yNew = gaussian_filter1d(yOld,sigma,truncate=trunc)
                if name in availableDataList:
                    index = availableDataList.index(name)
                    self.availableData[index][2] = yNew
                else:
                    self.availableData.append([name,xOld, yNew])

        
    def updatePlot(self):
        self.plotfield.removeItem(self.legend)
        self.plotfield.clear()
        self.legend = pg.LegendItem()
#        self.plotfield.addLegend()
        sel = list([str(item.text()) for item in self.dataList.selectedItems()])
        unite = self.plotDataList
        print('unite',unite)
        plotNow = []
        for elem in unite:
            for item in self.dataList.findItems(elem, QtCore.Qt.MatchExactly):
                order_new = item.info[3]
            if len(plotNow) != 0:
                for it in range(len(plotNow)):
                    for item in self.dataList.findItems(plotNow[it], QtCore.Qt.MatchExactly):
                        order = item.info[3]
                    if order_new <= order:
                        plotNow.insert(it,elem)
                        break
                    elif it == len(plotNow)-1:
                        plotNow.append(elem)
            else:
                plotNow.append(elem)
                        
        self.updateData(plotNow+sel)
        colorIndex = 0
        print('plotNow', plotNow)
        availableDataList = [item[0] for item in self.availableData]
        for item in self.p.param('Data in Plot').children():
            name = item.name()
            for item in self.dataList.findItems(name, QtCore.Qt.MatchExactly):
                visible = item.info[0]
                scale = item.info[4]
            index = availableDataList.index(name)
            if visible:
                c = self.plotfield.plot(self.availableData[index][1],self.availableData[index][2]*scale, pen=(colorIndex), name=name)
                self.legend.addItem(c, name=c.opts['name'])
                colorIndex += 1
        if sel[0] not in plotNow:
            index = availableDataList.index(sel[0])
            c = self.plotfield.plot(self.availableData[index][1],self.availableData[index][2], pen=(0.5), name=self.availableData[index][0])
            self.legend.addItem(c, name=c.opts['name'])
        axX=self.plotfield.getAxis('bottom')
        axY=self.plotfield.getAxis('left')
        xrange = axX.range
        yrange = axY.range

        self.legend.setPos(self.legend.mapFromItem(self.legend, QtCore.QPointF(xrange[0],yrange[1])))
        self.plotfield.showGrid(x=True, y=True, alpha=1)
        self.plotfield.autoRange(padding = 0)
        self.plotfield.addItem(self.legend)

    def setListItems(self):
        files = []
        groups = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.pathLogNr):
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(r, file))

        for name in files:
            if 'pose.smooth' in name:
                groups.append(['pose x', 0,1, name, True, 0.1, 1, 0, 1])
                groups.append(['pose y', 0,2, name, True, 0.1, 1, 0, 1])
                groups.append(['pose theta', 0,3, name, True, 0.1, 1, 0, 1])
            elif 'steer.put' in name:
                groups.append(['steer torque cmd', 0,2, name, True, 0.1, 1, 0, 1])
            elif 'steer.get' in name:
                groups.append(['steer torque eff', 0,5, name, True, 0.1, 1, 0, 1])
                groups.append(['steer position raw', 0,8, name, True, 0.1, 1, 0, 1])
            elif 'status.get' in name:
                groups.append(['steer position cal', 0,1, name, True, 0.1, 1, 0, 1])
            elif 'linmot.put' in name:
                groups.append(['brake position cmd', 0,1, name, True, 0.1, 1, 0, 1])
            elif 'linmot.get' in name:
                groups.append(['brake position effective', 0,1, name, True, 0.1, 1, 0, 1])
            elif 'rimo.put' in name:
                groups.append(['motor torque cmd left', 0,1, name, True, 0.1, 1, 0, 1])
                groups.append(['motor torque cmd right', 0,2, name, True, 0.1, 1, 0, 1]) 
            elif 'rimo.get' in name:
                groups.append(['motor rot rate left', 0,2, name, True, 0.1, 1, 0, 1])
                groups.append(['motor rot rate right', 0,9, name, True, 0.1, 1, 0, 1])
            elif 'vmu931' in name:
                groups.append(['accel x (forward)', 0,2, name, True, 70, 700, 0, 1])
                groups.append(['accel y (left)', 0,3, name, True, 70, 700, 0, 1])
                groups.append(['accel theta', 0,4, name, True, 5, 50, 0, 1])
            
        
        self.dataList.clear()
        groups.sort()
        rawDataNames = []
        for name, timeIndex, dataIndex, fileName, vis, sig, wid, order, scale in groups:
            rawDataNames.append(name)
            dataFrame = self.getCSV(fileName)
            xRaw = dataFrame.iloc[:,timeIndex]
            yRaw = dataFrame.iloc[:,dataIndex]
            item = QtGui.QListWidgetItem(name)
            item.setText(name)
            if name == 'pose theta':
                for i in range(len(yRaw)):
                    if yRaw[i] < -np.pi:
                        yRaw[i] = yRaw[i] + 2*np.pi
                    if yRaw[i] > np.pi:
                        yRaw[i] = yRaw[i] - 2*np.pi
            if name in ['accel x (forward)', 'accel y (left)', 'accel theta']:
                xRaw, yRaw = prep.interpolation(self,xRaw,yRaw,0.001)
            
            item.data = [xRaw, yRaw] #item.data = [x_data, y_data]
            item.info = [vis, sig, wid, order, scale]#item.info = [visible, filter_sigma, filter_width, order, scale]
            item = self.dataList.addItem(item)
#            self.availableData.append([name,xRaw, yRaw])
            
        if len(groups) == 16:
            print('Data status: complete')
        else:
            print('ACHTUNG! Missing Data!')
            
        #Add Preprocessed Data
        groups = []
        groups.append(['pose vx', ['pose x'], True, 1, 10, 1, 1])
        groups.append(['pose vy', ['pose y'], True, 1, 10, 1, 1])
        groups.append(['pose vtheta', ['pose theta'], True, 0.1, 1, 1, 1])
        groups.append(['pose ax', ['pose vx'], True, 0.1, 1, 2, 1])
        groups.append(['pose ay', ['pose vy'], True, 0.1, 1, 2, 1])
        groups.append(['pose atheta', ['pose vtheta'], True, 0.1, 1, 2, 1])
        groups.append(['vehicle slip angle', ['pose theta', 'pose vx', 'pose vy'], True, 0.1, 1, 2, 1])
        groups.append(['vehicle vx', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0.1, 1, 3, 1])
        groups.append(['vehicle vy', ['pose vx', 'pose vy', 'vehicle slip angle'], True, 0.1, 1, 3, 1])
        groups.append(['vehicle ax total', ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'vehicle vy'], True, 5, 50, 3, 1])
        groups.append(['vehicle ay total', ['pose theta', 'pose vtheta', 'pose vx', 'pose vy', 'vehicle slip angle', 'vehicle vx', 'vehicle vy'], True, 0.1, 1, 3, 1])
        groups.append(['vehicle ax only transl', ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'], True, 0.1, 1, 3, 1])
        groups.append(['vehicle ay only transl', ['pose theta', 'pose vx', 'pose vy', 'pose ax', 'pose ay'], True, 0.1, 1, 3, 1])
        
        self.needsPreprocessing = []
        for name, dep, vis, sig, wid, order, scale in groups:
            self.needsPreprocessing.append(name)
            item = QtGui.QListWidgetItem(name)
            item.setText(name)
            item.info = [vis, sig, wid, order, scale] #item.info = [visible, filter_sigma, filter_width, order, scale]
            item.dependencies = [dep]
            item = self.dataList.addItem(item)
        self.dataList.sortItems()
        
        self.updateData(rawDataNames)


    def getCSV(self, filePath):
        dataFrame = pd.read_csv(str(filePath))
        return dataFrame
    
    
    def getDirectories(self, path):
        folders = [f.name for f in os.scandir(path) if f.is_dir()]
        return folders

        
def main():
    app = QtGui.QApplication(sys.argv)
    _ = Clarity()
    app.exec_()


if __name__ == '__main__':
    main()