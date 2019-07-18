#!/usr/bin/env bash

#Created on Mon Apr  8 12:55:56 2019
#
#@author: mvb

#___user variables:
PREPROROOT='/home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets'
PREPROFOLDER=$(cd /home/mvb/0_ETH/01_MasterThesis/Logs_GoKart/LogData/DataSets; ls -t | head -1)
PREPROFOLDERPATH=$PREPROROOT/$PREPROFOLDER
echo $PREPROFOLDERPATH

SAVEPATH="/home/mvb/0_ETH/01_MasterThesis/SimData"
FOLDERNAME=$(date +"%Y%m%d-%H%M%S")
FOLDERPATH=$SAVEPATH/$FOLDERNAME
SIMTAG="_test"
SAVEFOLDERPATH=$SAVEPATH/$FOLDERNAME$SIMTAG
SIMLOGFILENAMES=()
PORT="6000"
#select files
ITER=0
FIRSTFILE=0
LASTFILE=10000
for i in $PREPROFOLDERPATH/*.pkl; do if ((ITER >= $FIRSTFILE && ITER <= $LASTFILE)); then SIMLOGFILENAMES+=($(basename "$i")); fi; ITER=$(( $ITER + 1)); done;
#SIMLOGFILENAMES=$(cd $PREPROFOLDERPATH; ls -l | egrep -v '^d')

VISUALIZATION=1
LOGGING=0

# Specify vehicle model to simulate with
# possible options: "mpc_dynamic", "hybrid_mlp", "hybrid_lstm"
VEHICLE_MODEL_TYPE="mpc_dynamic"
#VEHICLE_MODEL_TYPE="hybrid_mlp"
#VEHICLE_MODEL_TYPE="hybrid_lstm"

# Specify ML model name to be used for model
# only necessary for vehicle model types "hybrid_mlp" and "hybrid_lstm"
#VEHICLE_MODEL_NAME="5x64_relu_reg0p0" #for "hybrid_mlp" model
#VEHICLE_MODEL_NAME="5x64_relu_reg0p1" #for "hybrid_mlp" model
#VEHICLE_MODEL_NAME="2x32_relu_reg0p0" #for "hybrid_lstm" model
#VEHICLE_MODEL_NAME="2x32_relu_reg0p01" #for "hybrid_lstm" model
#VEHICLE_MODEL_NAME="1x32_relu_reg0p01" #for "hybrid_lstm" model
VEHICLE_MODEL_NAME="2x32_tanh_reg0p1" #for "hybrid_lstm" model

python3 kartsim_server.py $PORT $VISUALIZATION $LOGGING $VEHICLE_MODEL_TYPE $VEHICLE_MODEL_NAME &
SRVPID=$!

if [ $VISUALIZATION = 1 ]
then
    python3 kartsim_visualizationclient.py $PORT &
    VIZPID=$!
fi

if [ $LOGGING = 1 ]
then
    mkdir $SAVEFOLDERPATH
    for i in "${SIMLOGFILENAMES[@]}"; do cp $PREPROFOLDERPATH/"${i}" $SAVEFOLDERPATH; done

    python3 kartsim_loggerclient.py $PORT $SAVEFOLDERPATH $VEHICLE_MODEL_TYPE $VEHICLE_MODEL_NAME "${SIMLOGFILENAMES[@]}" &
    LOGPID=$!
fi


#python3 joystick/joystick_client.py &
#python3 user/simtompc_comparison_client.py &
python3 user/real_time_sim_client.py $PORT $LOGGING $SAVEFOLDERPATH $PREPROFOLDERPATH "${SIMLOGFILENAMES[@]}" &
#python3 user/dummyClient.py &
SIMPID=$!

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