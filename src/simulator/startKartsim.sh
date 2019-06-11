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
#SIMTAG="_MM_original_eval"
SIMTAG="_test"
SAVEFOLDERPATH=$SAVEPATH/$FOLDERNAME$SIMTAG
SIMLOGFILENAMES=()

#select files
ITER=0
FIRSTFILE=0
LASTFILE=10000
for i in $PREPROFOLDERPATH/*.pkl; do if ((ITER >= $FIRSTFILE && ITER <= $LASTFILE)); then SIMLOGFILENAMES+=($(basename "$i")); fi; ITER=$(( $ITER + 1)); done;
#SIMLOGFILENAMES=$(cd $PREPROFOLDERPATH; ls -l | egrep -v '^d')
#echo $SIMLOGFILENAMES
VISUALIZATION=1
LOGGING=1

mkdir $SAVEFOLDERPATH

for i in "${SIMLOGFILENAMES[@]}"; do cp $PREPROFOLDERPATH/"${i}" $SAVEFOLDERPATH; done

python3 kartsim_server.py $VISUALIZATION $LOGGING &
SRVPID=$!

if [ $VISUALIZATION = 1 ]
then
    python3 kartsim_visualizationclient.py &
    VIZPID=$!
fi

if [ $LOGGING = 1 ]
then
    python3 kartsim_loggerclient.py $SAVEFOLDERPATH "${SIMLOGFILENAMES[@]}" &
    LOGPID=$!
fi


#python3 joystick/joystick_client.py &
#python3 user/simtompc_comparison_client.py &
python3 user/evaluationClient.py $SAVEFOLDERPATH $PREPROFOLDERPATH "${SIMLOGFILENAMES[@]}" &
#python3 user/dummyClient.py &
SIMPID=$!
echo $SIMPID

exit_script() {
    echo " "
    echo "Kill server, visualization and logger"
    trap - INT # clear the trap
    kill -9 ${SRVPID}
    if [ $VISUALIZATION = 1 ]
    then
        echo "Kill viz"
        kill -9 ${VIZPID}
    fi

    if [ $LOGGING = 1 ]
    then
        echo "Kill log"
        kill -9 ${LOGPID}
    fi
    echo "Kill sim"
    kill -9 ${SIMPID}
}

trap exit_script INT
sleep infinity