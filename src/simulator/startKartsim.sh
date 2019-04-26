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
SIMTAG="_test"
SAVEFOLDERPATH=$SAVEPATH/$FOLDERNAME$SIMTAG
SIMLOGFILENAMES=()

#select files
ITER=0
FIRSTFILE=2
LASTFILE=2
for i in $PREPROFOLDERPATH/*.pkl; do if ((ITER >= $FIRSTFILE && ITER <= $LASTFILE)); then SIMLOGFILENAMES+=($(basename "$i")); fi; ITER=$(( $ITER + 1)); done;
#SIMLOGFILENAMES=$(cd $PREPROFOLDERPATH; ls -l | egrep -v '^d')
#echo $SIMLOGFILENAMES
VISUALIZATION=1
LOGGING=1

mkdir $SAVEFOLDERPATH
#for i in $SIMLOGFILENAMES; do cp $PREPROFOLDERPATH/$i $SAVEFOLDERPATH; done
#IFS=', ' read -r -a array <<< "$SIMLOGFILENAMES"
#echo "${array[@]}"
#for i in "${array[@]}"; do echo "${i}"; echo "p"; done
for i in "${SIMLOGFILENAMES[@]}"; do cp $PREPROFOLDERPATH/"${i}" $SAVEFOLDERPATH; done

python basicKartsimServer.py $VISUALIZATION $LOGGING &
SRVPID=$!

if (( $VISUALIZATION ))
then
    python basicVisualizationClient.py &
    VIZPID=$!
fi

python basicLoggerClient.py $SAVEFOLDERPATH "${SIMLOGFILENAMES[@]}" &
LOGPID=$!

#python basicSimulationClient.py $SAVEFOLDERPATH $PREPROFOLDERPATH &
python evaluationClient.py $SAVEFOLDERPATH $PREPROFOLDERPATH "${SIMLOGFILENAMES[@]}" &
#echo $SAVEFOLDERPATH $PREPROFOLDERPATH "${SIMLOGFILENAMES[@]}"
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