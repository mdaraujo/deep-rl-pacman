# Deep Reinforcement Learning in Pac-Man

Training Deep Reinforcement Learning agents in a custom Gym environment adapted from a Client-Server Pac-Man clone.  

Trained agents video: https://www.youtube.com/watch?v=xDTF0w38WR0

## Prerequisites

    Python >= 3.5
    Stabe-Baselines==2.10.0

## Usage

### Train Example

```
python3 rl_train.py -t 1000000 -a PPO -g 0.92 -pr
```

### Test Example

```
python3 rl_test.py agents/10_1008_PPO_982_PPO_Mult_Pos_5M_0.92_201218-130851/
```

### Run Client

```
python3 rl_client.py agents/10_1008_PPO_982_PPO_Mult_Pos_5M_0.92_201218-130851/
```

## Client-Server Pac-Man Instructions
Client-Server Pac-Man clone

![Demo](https://github.com/dgomes/iia-ia-pacman/raw/master/data/Screenshot%202019-11-14%20at%2015.55.22.png)

### Install

* Clone this repository
* Create a virtual environment:

```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
### How to run:
Open 3 terminals, in each terminal run once:
```console
$ source venv/bin/activate
```
Run each application in its terminal:

Terminal 1:
```console
$ python server.py
```
Terminal 2:
```console
$ python viewer.py
```
Terminal 3:
```console
$ python client.py
```

### Credits
Sprites from https://github.com/rm-hull/big-bang/tree/master/examples/pacman/data
