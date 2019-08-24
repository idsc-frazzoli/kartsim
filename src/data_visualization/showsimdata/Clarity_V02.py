#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:38:24 2019

@author: mvb
"""
from config import directories
import data_visualization.data_io as dio
from file_grave_yard.gokart_data_processing_oldfiles.preprocessing import updateData

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys
import os
import copy

from pyqtgraph.dockarea import DockArea, Dock
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes

class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, name):
        self.addChild(
            dict(name="%s" % (str(name)), type='bool', value=True, removable=True, renamable=True,
                 expanded=False))


class Clarity(QtGui.QMainWindow):

    def __init__(self):
        super(Clarity, self).__init__()
#        self.pathRootData = '/home/mvb/0_ETH/01_MasterThesis/SimData/20190411-154017'
        self.pathRootSimData = os.path.join(directories['root'][:-14], 'SimData', 'lookatlogs')
        # self.pathRootSimData = os.path.join(directories['root'], 'Data', 'MLPDatasets')
        # self.pathRootSimData = os.path.join(directories['root'], 'Data', 'Sampled')
        # self.pathRootSimData = os.path.join(directories['root'], 'Evaluation')
        simFolders = dio.getDirectories(self.pathRootSimData)
        simFolders.sort()
        defaultSim = simFolders[-1]
        defaultSim = 'test'
        # defaultSim = '20190729-115111_test_normal/test_log_files'
        self.pathSimData = os.path.join(self.pathRootSimData, defaultSim)
        print('Loading data from', self.pathSimData)

        logNrs = []
        for r, d, f in os.walk(self.pathSimData):
            for file in f:
                if '.pkl' in file:
                    logNrs.append(file[:-19])

        defaultLogNr = logNrs[0]
        self.prefix = defaultLogNr

        params = [
#            {'name': 'Testing day', 'type': 'list', 'values': testDays, 'value': defaultDay},
           {'name': 'Log Nr.', 'type': 'list', 'values': logNrs, 'value': defaultLogNr},
#            {'name': 'Add to plot -->', 'type': 'action'},
            ScalableGroup(name='Data in plot'),
            ScalableGroup(name='Data in scatter plot'),
        ]
        self.p = Parameter.create(name='params', type='group', children=params)
#        self.p.param('Testing day').sigValueChanged.connect(self.testDayChange)
        self.p.param('Log Nr.').sigValueChanged.connect(self.logNrChange)
#        self.p.param('Add to plot -->').sigActivated.connect(self.addToPlot)
        self.p.param('Data in plot').sigTreeStateChanged.connect(self.treeChange)
        self.p.param('Data in scatter plot').sigTreeStateChanged.connect(self.treeChange)

        self.dataList = QtGui.QListWidget()
        self.dataList.setSelectionMode(self.dataList.ExtendedSelection)
        self.dataList.itemSelectionChanged.connect(self.dataSelectionChanged)
        
        self.plotfield = pg.PlotWidget()
        
        self.scatterfield = pg.PlotWidget()
        self.scatterData = {}
        
        self.histogramfield = pg.PlotWidget()
        
        self.plotLegend = pg.LegendItem()
        self.scatterLegend = pg.LegendItem()
        
        self.tree = ParameterTree(showHeader=False)
        self.tree.setParameters(self.p, showTop=False)
#        self.tree = ParameterTree(showHeader=False)
#        self.tree.setParameters(self.p, showTop=False)
        
        self.btn_addToPlot = QtGui.QPushButton('Add to Plot')
        self.btn_addToPlot.clicked.connect(self.addToPlot)
        
        self.btn_addToScatter = QtGui.QPushButton('Add to Scatter Plot:\nSelect two signals!')
        self.btn_addToScatter.clicked.connect(self.addToScatter)
        self.btn_addToScatter.setEnabled(False)
        
        self.btn_clear_plots = QtGui.QPushButton('Clear plots')
        self.btn_clear_plots.clicked.connect(self.clearPlots)

        self.availableData = []
        
        self.createUI()

    def createUI(self):
        self.dockArea = DockArea()
        self.setCentralWidget(self.dockArea)
        windowWidth = 1200
        windowHeight = 675
        self.setGeometry(100, 100, windowWidth, windowHeight)
        self.setWindowTitle('Clarity')

        pg.setConfigOptions(antialias=True)

        d1 = Dock("Display Data", size=(windowWidth / 3., windowHeight / 3.))
        d2 = Dock("Plot 1", size=(windowWidth * 2 / 3., windowHeight*2./3.))
        d3 = Dock("Parameter Tree", size=(windowWidth / 3., windowHeight* 3/ 6.))
        d4 = Dock("Histogram 1", size=(windowWidth * 2 / 3., windowHeight/3.))
        d5 = Dock("Buttons", size=(windowWidth / 3., windowHeight/6.))

        self.dockArea.addDock(d1, 'left')
        self.dockArea.addDock(d2, 'right')
        self.dockArea.addDock(d3, 'bottom', d1)
        self.dockArea.addDock(d4, 'bottom', d2)
        self.dockArea.addDock(d5, 'bottom', d1)
        
        d1.addWidget(self.dataList)

        d2.addWidget(self.plotfield)

        self.setListItems()
        d3.addWidget(self.tree)
        
        d4.addWidget(self.histogramfield)
        
        d5.addWidget(self.btn_addToPlot,0,0)
        d5.addWidget(self.btn_addToScatter,0,1)
        d5.addWidget(self.btn_clear_plots,1,0)
        

        self.show()
    
    def clearPlots(self):
        self.scatterfield.clear()
        self.plotfield.clear()
        self.p.param('Data in plot').clearChildren()
        self.p.param('Data in scatter plot').clearChildren()
    
    def filterChange(self, group, param):
        print('filter changed!')
        for param, value, data in param:
            if value != 'parent':
                if param.name() == 'sigma':
                    self.kartData[group.name()]['info'][1] = data
                    self.kartData[group.name()]['info'][2] = 10 * data
                if param.name() == 'scale':
                    self.kartData[group.name()]['info'][3] = data
                self.updatePlot()

#    def testDayChange(self, param):
#        self.plotfield.clear()
#        self.p.param('Data in plot').clearChildren()
#        self.plottedData = []
#
#        value = param.value()
#        self.pathTestDay = self.pathRootData + '/' + value
#        logNrs = dio.getDirectories(self.pathTestDay)
#        logNrs.sort()
#        self.p.param('Log Nr.').remove()
#        child = Parameter.create(name='Log Nr.', type='list', values=logNrs, value=logNrs[0])
#        self.p.insertChild(1, child)
#        self.p.param('Log Nr.').sigValueChanged.connect(self.logNrChange)
#        self.pathLogNr = self.pathTestDay + '/' + logNrs[0]
#
    def logNrChange(self, param):
       self.p.param('Data in plot').clearChildren()
       self.p.param('Data in scatter plot').clearChildren()

       value = param.value()
       self.prefix = value
       self.setListItems()

    def treeChange(self, group, param):
        print('Tree changed!')
        for param, value, data in param:
            if value != 'parent':
                if param.name() == 'Data in plot':
                    if value != 'childRemoved':
                        self.updatePlot()
                if param.name() == 'Data in scatter plot':
                    if value != 'childRemoved':
                        self.updatePlot()
                elif value == 'value':
                    for item in self.dataList.findItems(param.name(), QtCore.Qt.MatchExactly):
                        self.kartData[param.name()]['info'][0] = data
                    self.updatePlot()

    def dataSelectionChanged(self):
        print('dataselection changed')
        selected = self.dataList.selectedItems()
        
        # if len(selected) > 2:
        #     self.dataList.blockSignals(True)
        #     try:
        #         for item in selected[1:-1]:
        #             item.setSelected(False)
        #     finally:
        #         self.dataList.blockSignals(False)
        if len(selected) == 2:
            self.btn_addToScatter.setEnabled(True)
            self.btn_addToScatter.setText('Add to Scatter Plot')
        else:
            self.btn_addToScatter.setEnabled(False)
            self.btn_addToScatter.setText('Add to Scatter Plot:\nSelect two signals!')
        if len(selected) == 1:
            self.updatePlot()
        

    def addToScatter(self):
        sel = list([str(item.text()) for item in self.dataList.selectedItems()])
        dispName = 0
        
        if len(sel) == 1:
            print('press \'Add to plot\'!')
            return
        if len(sel) == 2:
            dispName = str(sel[0]) + ' / ' + str(sel[1])
            self.scatterData[dispName] = {}
            self.scatterData[dispName]['data'] = [self.kartData[sel[0]]['data'][1], self.kartData[sel[1]]['data'][1]]
            self.scatterData[dispName]['info'] = [[1],[0,0,1],[0,0,1]]
        
        if not 'Scatter Plot' in self.dockArea.findAll()[1]:
            print('exist')
            d6 = Dock("Scatter Plot")
            self.dockArea.addDock(d6, 'above', self.dockArea.findAll()[1]['Plot 1'])
            d6.addWidget(self.scatterfield)
        if (not (any([item.name() == dispName for item in
                      self.p.param('Data in scatter plot').children()])) and dispName != 0):
            self.p.param('Data in scatter plot').addNew(dispName)
            sigma1 = self.kartData[sel[0]]['info'][1]
            scale1 = self.kartData[sel[0]]['info'][3]
            sigma2 = self.kartData[sel[1]]['info'][1]
            scale2 = self.kartData[sel[1]]['info'][3]
            OptionVars = [
                {'name': 'sigma '+str(sel[0]), 'type': 'float', 'value': sigma1, 'step': 0.1},
                {'name': 'scale '+str(sel[0]), 'type': 'float', 'value': scale1, 'step': 0.1},
                {'name': 'sigma '+str(sel[1]), 'type': 'float', 'value': sigma2, 'step': 0.1},
                {'name': 'scale '+str(sel[1]), 'type': 'float', 'value': scale2, 'step': 0.1},
            ]
            #            child = Parameter.create(name='Gaussian Filter', type='group', children
            #            = filterVars)
            self.p.param('Data in scatter plot').param(dispName).addChildren(OptionVars)
            self.p.param('Data in scatter plot').param(dispName).sigTreeStateChanged.connect(
                self.filterChange)
        elif dispName == 0:
            print('No Data Selected!')
        else:
            print(dispName + ' is already plotted!')
            

    def addToPlot(self):
        sel = list([str(item.text()) for item in self.dataList.selectedItems()])
        # dispName = 0
        # if len(sel) == 1:
        #     dispName = sel[0]
        # if len(sel) == 2:
        #     print('press \'Add to scatter plot\'!')
        if len(sel) == 0:
            print('No Data Selected!')
        for dispName in sel:
            if (not (any([item.name() == dispName for item in
                          self.p.param('Data in plot').children()])) and dispName != 0):
                self.p.param('Data in plot').addNew(dispName)
                sigma = self.kartData[dispName]['info'][1]
                scale = self.kartData[dispName]['info'][3]
                OptionVars = [
                    {'name': 'sigma', 'type': 'float', 'value': sigma, 'step': 0.1},
                    {'name': 'scale', 'type': 'float', 'value': scale, 'step': 0.1},
                ]
                #            child = Parameter.create(name='Gaussian Filter', type='group', children
                #            = filterVars)
                self.p.param('Data in plot').param(dispName).addChildren(OptionVars)
                self.p.param('Data in plot').param(dispName).sigTreeStateChanged.connect(
                    self.filterChange)
            elif len(sel) == 0:
                print('No Data Selected!')
            else:
                print(dispName + ' is already plotted!')
            

    def updatePlot(self):
        self.plotfield.removeItem(self.plotLegend)
        self.plotfield.clear()
        self.histogramfield.clear()
        self.plotLegend = pg.LegendItem()
        
        self.scatterfield.removeItem(self.scatterLegend)
        self.scatterfield.clear()
        self.histogramfield.clear()
        self.scatterLegend = pg.LegendItem()
        #        self.plotfield.addLegend()
        sel = list([str(item.text()) for item in self.dataList.selectedItems()])
        
        kartDataMod = updateData(copy.deepcopy(self.kartData),self.dataNames)
        
#        self.filter_raw_data()
        colorIndex = 0
        plotNames = []
        for item in self.p.param('Data in plot').children():
            name = item.name()
            plotNames.append(name)
            visible = kartDataMod[name]['info'][0]
            scale = kartDataMod[name]['info'][3]
            if visible:
                c = self.plotfield.plot(kartDataMod[name]['data'][0], 
                                        kartDataMod[name]['data'][1] * scale, pen=(colorIndex),
                                        name=name)
            #histogram
                y,x = np.histogram(kartDataMod[name]['data'][1], bins=100)
                _ = self.histogramfield.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),name=name)
                
                self.plotLegend.addItem(c, name=c.opts['name'])
                colorIndex += 1
        
        colorIndex = 0
        for item in self.p.param('Data in scatter plot').children():
            dispName = item.name()
            name1,name2= dispName.split('/')
            visible = self.scatterData[dispName]['info'][0][0]
            scale1 = self.scatterData[dispName]['info'][1][2]
            scale2 = self.scatterData[dispName]['info'][2][2]
            if visible:
                c = self.scatterfield.plot(self.scatterData[dispName]['data'][0] * scale1, 
                                        self.scatterData[dispName]['data'][1] * scale2, pen=pg.mkPen(colorIndex, width=2),
                                        name=dispName)
                
                self.scatterLegend.addItem(c, name=c.opts['name'])
                colorIndex += 1
        if sel[0] not in plotNames:
            c = self.plotfield.plot(kartDataMod[sel[0]]['data'][0], kartDataMod[sel[0]]['data'][1],
                                    pen=(0.5), name=sel[0])
        #histogram 
            y,x = np.histogram(kartDataMod[sel[0]]['data'][1], bins=100)
            _ = self.histogramfield.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150),name=sel[0])
            self.histogramfield.addLine(x = np.mean(kartDataMod[sel[0]]['data'][1]), y = None, pen=pg.mkPen('r', width=3))
            self.histogramfield.addLine(x = np.mean(kartDataMod[sel[0]]['data'][1]) +np.std(kartDataMod[sel[0]]['data'][1]) , y = None, pen=pg.mkPen(color = (200,200,200), width=1))
            self.histogramfield.addLine(x = np.mean(kartDataMod[sel[0]]['data'][1]) -np.std(kartDataMod[sel[0]]['data'][1]) , y = None, pen=pg.mkPen(color = (200,200,200), width=1))
            
            self.plotLegend.addItem(c, name=c.opts['name'])
        axX = self.plotfield.getAxis('bottom')
        axY = self.plotfield.getAxis('left')
        axsX = self.scatterfield.getAxis('bottom')
        axsY = self.scatterfield.getAxis('left')
        xrange1 = axX.range
        yrange1 = axY.range
        xrange2 = axsX.range
        yrange2 = axsY.range

        self.plotLegend.setPos(
            self.plotLegend.mapFromItem(self.plotLegend, QtCore.QPointF(xrange1[0], yrange1[1])))
        self.plotfield.showGrid(x=True, y=True, alpha=1)
        self.plotfield.autoRange(padding=0.1)
        self.plotfield.addItem(self.plotLegend)
        
        
#        self.scatterLegend.updateSize()
        self.scatterfield.showGrid(x=True, y=True, alpha=1)
        self.scatterfield.autoRange(padding=0.1)
        self.scatterfield.setAspectLocked(True)
        self.scatterLegend.setPos(
            self.scatterLegend.mapFromItem(self.scatterLegend, QtCore.QPointF(xrange2[0], yrange2[1])))
        self.scatterfield.addItem(self.scatterLegend)


    def setListItems(self):
        self.dataList.clear()
        csvFiles = []
        pklFiles = []
        rawData = {}
        self.kartData = {}
        for r, d, f in os.walk(self.pathSimData):
            for file in f:
                if '.csv' in file and self.prefix in file:
                    csvFiles.append([os.path.join(r, file),file])
                if '.pkl' in file and self.prefix in file:
                    pklFiles.append([os.path.join(r, file),file])
            break

        self.dataNames = []
        timeTopic = ''
        for csvfile, csvfileName in csvFiles:
            try:
                dataFrame = dio.dataframe_from_csv(csvfile)
                for topic in dataFrame.columns:
                    if 'time' in topic or 'Time' in topic:
                        timeTopic = topic
                        break
                if len(timeTopic) != 0:
                    for topic in dataFrame.columns:
                        if topic != timeTopic:
                            if not 'MH' in topic and not 'vehicle' in topic and 'pose' not in topic:
                                self.dataNames.append(topic)
                            rawData[topic] = {}
                            rawData[topic]['data'] = [dataFrame[timeTopic].values, dataFrame[topic].values]
                            rawData[topic]['info'] = [1, 0, 0, 1]
                else:
                    print('No column name called \'time\' or \'Time\' found in dataset.')
                    raise ValueError
            except:
                print('Could not read data from file ', csvfile)
                raise

            if 'pose atheta [rad*s^-2]' not in rawData.keys():
                preproTopics = ['pose vx [m*s^-1]', 'pose vy [m*s^-1]', 'pose ax [m*s^-2]',
                                      'pose ay [m*s^-2]', 'pose atheta [rad*s^-2]', 'vehicle slip angle [rad]',]
                                      # 'vehicle ax total', 'vehicle ay total',
                                      # 'vehicle ax only transl', 'vehicle ay only transl']
                for topic in preproTopics:
                    rawData[topic] = {}
                    rawData[topic]['data'] = [[],[]]
                    rawData[topic]['info'] = [1, 0, 0, 1]
                    self.dataNames.append(topic)

                try:
                    rawData = updateData(rawData,self.dataNames)
                except KeyError:
                    print('Some key missing. No preprocessing possible')

            for topic in rawData:
                if timeTopic:
                    topicNew = topic + ' [' + csvfileName[:-4] + ']'
                    self.kartData[topicNew] = {}
                    self.kartData[topicNew]['data'] = [rawData[topic]['data'][0], rawData[topic]['data'][1]]
                    self.kartData[topicNew]['info'] = [rawData[topic]['info'][0],
                                                      rawData[topic]['info'][1],
                                                      rawData[topic]['info'][2],
                                                      rawData[topic]['info'][3]]

                    item = QtGui.QListWidgetItem(topicNew)
                    item.setText(topicNew)
                    item = self.dataList.addItem(item)
        
        self.dataNames = []
        rawData = {}
        for pklfile , pklfileName in pklFiles:
            try:
                dataFrame = dio.getPKL(pklfile)
                for topic in dataFrame.columns:
                    if 'time' in topic or 'Time' in topic:
                        timeTopic = topic
                        break
                for topic in dataFrame.columns:
                    if topic != timeTopic:
                        if 'MH' not in topic and 'vehicle' not in topic and 'vmu' not in topic and 'pose' not in topic:
                            self.dataNames.append(topic)
                        rawData[topic] = {}
                        rawData[topic]['data'] = [dataFrame[timeTopic].values, dataFrame[topic].values]
                        rawData[topic]['info'] = [1, 0, 0, 1]
            except:
                print('EmptyDataError: could not read data from file ', pklfile)
                raise

            rawData = updateData(rawData,self.dataNames)

            for topic in rawData:
                if timeTopic:
                    topicNew = topic + ' [' + pklfileName[:-4] + ']'
                    self.kartData[topicNew] = {}
                    self.kartData[topicNew]['data'] = [rawData[topic]['data'][0], rawData[topic]['data'][1]]
                    self.kartData[topicNew]['info'] = [rawData[topic]['info'][0],
                                                      rawData[topic]['info'][1],
                                                      rawData[topic]['info'][2],
                                                      rawData[topic]['info'][3]]

                    item = QtGui.QListWidgetItem(topicNew)
                    item.setText(topicNew)
                    item = self.dataList.addItem(item)

        self.dataList.sortItems()
        self.dataNames = []
        for key in self.kartData:
            self.dataNames.append(key)
            
#        self.filter_raw_data()


def main():
    app = QtGui.QApplication(sys.argv)
    _ = Clarity()
    app.exec_()


if __name__ == '__main__':
    main()
