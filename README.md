# twitchchess

A toy implementation of neural network chess written while livestreaming.

Stream
-----

https://www.twitch.tv/tomcr00s3

Usage
-----

```
 ./play.py   # runs webserver on localhost:5000
```

Implementation
-----

twitchchess is a simple 1 look ahead neural network value function. The trained net is in nets/value.pth. It takes in a serialized board and outputs a range from -1 to 1. -1 means black is win, 1 means white is win.

Serialization
-----

We serialize the board into a 8x8x5 bitvector. See state.py for how.

Training Set
-----

The value function was trained on 5M board positions from http://www.kingbase-chess.net/

