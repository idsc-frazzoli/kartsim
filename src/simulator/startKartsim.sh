#!/usr/bin/env bash

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
SIMLOGFILENAMES=()
for i in $PREPROFOLDERPATH/*.pkl; do SIMLOGFILENAMES+="$(basename "$i"),"; done;
#SIMLOGFILENAMES=$(cd $PREPROFOLDERPATH; ls -l | egrep -v '^d')
echo $SIMLOGFILENAMES
#VISUALIZATION=true
VISUALIZATION=false

mkdir $SAVEFOLDERPATH

cp $PREPROFOLDERPATH/* $SAVEFOLDERPATH

python basicKartsimServer.py $VISUALIZATION &
SRVPID=$!
if [ $VISUALIZATION = true ]
then
    python basicVisualizationClient.py &
    VIZPID=$!
fi
python basicLoggerClient.py $SAVEFOLDERPATH "${SIMLOGFILENAMES[@]}" &
LOGPID=$!
python basicSimulationClient.py $SAVEFOLDERPATH $PREPROFOLDERPATH &
#python validationClient.py $SAVEFOLDERPATH $PREPROFOLDERPATH &
SIMPID=$!

exit_script() {
    echo " "
    echo "Kill server, visualization and logger"
    trap - INT # clear the trap
    kill ${SRVPID}
    if [ $VISUALIZATION = true ]
    then
        kill ${VIZPID}
    fi
    kill ${LOGPID}

    kill ${SIMPID}

}

trap exit_script INT
sleep infinity