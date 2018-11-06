# twitchchess

A toy implementation of ~~neural network~~ normal chess written while livestreaming.

<img width=600px src="https://raw.githubusercontent.com/geohot/twitchchess/master/screenshot.png" />

Can beat me at bullet, maybe rated 1200-1400. Search is 3-ply full with a 5-ply beam(x10).

Nice TODO
-----

* Value function in play.py/ClassicValuator, plenty of room for improvement.
* Add Quiescence search to play decent endgame 
* Is there a bug which allows draws to happen?

Stream
-----

https://www.twitch.tv/tomcr00s3

Usage
-----

```
 pip3 install python-chess torch torchvision numpy flask
 # then...
 ./play.py   # runs webserver on localhost:5000
```

Or with pypy (for max speed)
```
 pip_pypy install python-chess flask
 pypy ./play.py
 # web browse to localhost:5000
```

TODOs
-----

* Roll out search beyond 1-ply
* Make trainer multi GPU
* Train on more data
* Add RL self play learning support

Implementation
-----

twitchchess is a simple 1 look ahead neural network value function. The trained net is in nets/value.pth. It takes in a serialized board and outputs a range from -1 to 1. -1 means black is win, 1 means white is win.

Serialization
-----

We serialize the board into a 8x8x5 bitvector. See state.py for how.

Training Set
-----

The value function was trained on 5M board positions from http://www.kingbase-chess.net/

