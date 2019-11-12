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
SIMLOGFILENAMES=()
PORT="6000"

#select files
ITER=0
FIRSTFILE=0
LASTFILE=10000
for i in $PREPROFOLDERPATH/*.pkl; do if ((ITER >= $FIRSTFILE && ITER <= $LASTFILE)); then SIMLOGFILENAMES+=($(basename "$i")); fi; ITER=$(( $ITER + 1)); done;

VISUALIZATION=1
LOGGING=0

# Specify vehicle model type and the model name to simulate with
# possible options: "mpc_dynamic", "hybrid_mlp",
VEHICLE_MODEL_TYPE="mpc_dynamic"
VEHICLE_MODEL_NAME="original"

#VEHICLE_MODEL_TYPE="mpc_kinematic"
#VEHICLE_MODEL_NAME="original"

#VEHICLE_MODEL_TYPE="hybrid_mlp"
# VEHICLE_MODEL_NAME="0x6_None_reg0p0_dyn_symmetric" #for "hybrid_mlp" model
#VEHICLE_MODEL_NAME="1x16_tanh_reg0p0_dyn_symmetric" #for "hybrid_mlp" model
# VEHICLE_MODEL_NAME="2x16_softplus_reg0p01_dyn_symmetric" #for "hybrid_mlp" model

#VEHICLE_MODEL_TYPE="hybrid_kinematic_mlp"
# VEHICLE_MODEL_NAME="1x16_tanh_reg0p0_kin_symmetric" #for "hybrid_kinematic_mlp" model
# VEHICLE_MODEL_NAME="2x16_tanh_reg0p0_kin_symmetric" #for "hybrid_kinematic_mlp" model
# VEHICLE_MODEL_NAME="3x32_tanh_reg0p0_kin_symmetric" #for "hybrid_kinematic_mlp" model
# VEHICLE_MODEL_NAME="4x64_tanh_reg0p0_kin_symmetric" #for "hybrid_kinematic_mlp" model

#VEHICLE_MODEL_TYPE="no_model"
# VEHICLE_MODEL_NAME="1x16_tanh_reg0p0_kin_symmetric" #for "no_model" model
# VEHICLE_MODEL_NAME="2x16_tanh_reg0p0_kin_symmetric" #for "no_model" model
# VEHICLE_MODEL_NAME="3x32_tanh_reg0p0_kin_symmetric" #for "no_model" model
# VEHICLE_MODEL_NAME="4x64_tanh_reg0p0_kin_symmetric" #for "no_model" model


SIMTAG=_$VEHICLE_MODEL_TYPE\_$VEHICLE_MODEL_NAME
SAVEFOLDERPATH=$SAVEPATH/$FOLDERNAME$SIMTAG

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
#    for i in "${SIMLOGFILENAMES[@]}"; do cp $PREPROFOLDERPATH/"${i}" $SAVEFOLDERPATH; done

    python3 kartsim_loggerclient.py $PORT $SAVEFOLDERPATH $VEHICLE_MODEL_TYPE $VEHICLE_MODEL_NAME "${SIMLOGFILENAMES[@]}" &
    LOGPID=$!
fi

#python3 joystick/joystick_client.py &
#python3 user/simtompc_comparison_client.py &
#python3 user/real_time_sim_client.py $PORT $LOGGING $SAVEFOLDERPATH $PREPROFOLDERPATH "${SIMLOGFILENAMES[@]}" &
#python3 user/dummyClient.py &
#SIMPID=$!

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