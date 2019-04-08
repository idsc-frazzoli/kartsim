#!/bin/sh

#Created on Mon Apr  8 12:55:56 2019
#
#@author: mvb


python pyKartsimServer.py &

python basicVisualizationClient.py &

python basicLoggerClient.py &

exit_script() {
    echo " "
    echo "Kill server, visualization and logger"
    trap - INT # clear the trap
    kill -- -$$ # Sends SIGTERM to child/sub processes
}

trap exit_script INT
sleep infinity