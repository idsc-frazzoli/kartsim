# Launching Kartsim

- Go to folder "simulator" and execute: 
```ruby 
./startKartsim.sh
``` 
This will start a server executing the simulation with the selected, as well as  a client for logging the data generated during simulation and a second client for the visualization during simulation.

- You can then send a message to the server containing the current state of the vehicle including the inputs to the system and a time step defining the simulation duration. An example for such a client in python is shown in dummyClient.py and can be launched from the "user" folder in your terminal using:
```ruby
python dummyClient.py
```
