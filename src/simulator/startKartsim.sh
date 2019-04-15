#!/bin/sh

#Created on Mon Apr  8 12:55:56 2019
#
#@author: mvb

#___user variables:
PREPROROOT='/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'
PREPROFOLDER=$(cd /home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets; ls -t | head -1)
PREPROFOLDERPATH=$PREPROROOT/$PREPROFOLDER

SAVEPATH="/home/mvb/0_ETH/01_MasterThesis/SimData"
FOLDERNAME=$(date +"%Y%m%d-%H%M%S")
FOLDERPATH=$SAVEPATH/$FOLDERNAME
SIMTAG="_validation"
SAVEFOLDERPATH=$SAVEPATH/$FOLDERNAME$SIMTAG
mkdir $SAVEFOLDERPATH

cp $PREPROFOLDERPATH/* $SAVEFOLDERPATH

python basicKartsimServer.py &
SRVPID=$!
python basicVisualizationClient.py &
VIZPID=$!
python basicLoggerClient.py $SAVEFOLDERPATH $SIMTAG &
LOGPID=$!
#python basicSimulationClient.py &
python validationClient.py $PREPROFOLDERPATH &
SIMPID=$!
exit_script() {
    echo " "
    echo "Kill server, visualization and logger"
    trap - INT # clear the trap
    kill ${SRVPID}
    kill ${VIZPID}
    kill ${LOGPID}
    kill ${SIMPID}

}

trap exit_script INT
sleep infinity