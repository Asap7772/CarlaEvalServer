# CarlaEvalServer

### About
Created Flask Server to enable the evaluation environment to work with multiple experiments.
I use an md5 hash to compare the file hashes of pickled torch policies to see if policy was changed.
If so, a rollout of the policy was done on the evaluation environment.
One can add paths to the pickle files for the currently running experiments. 
It then sequentially evaluates them. 
When you stop the server, it saves the average return over time + plots.

### Dependencies
- Rail-rl-private
- Carla
- Flask
