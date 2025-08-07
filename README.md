# Traffic Simulation
 
Fun little traffic simulation with a traffic light.

Driver reaction times, accelerations, and braking power are all drawn randomly from a normal distribution for a little bit of extra realism.
The speed limit is calculated with a bit of simple physics too.

Traffic jams are highlighted in red on the density plot.

Additionally, a velocity plot is shown too.

These are calculated using a moving average with a window size of twice the following distance.

![traffic simulation gif](https://github.com/Turtlely/traffic/blob/ad2969311a98010e88876234305654b1988f6295/traffic_simulation.gif)
