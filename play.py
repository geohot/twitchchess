#!/usr/bin/env python3
import chess
import torch
from state import State
from train import Net

class Valuator(object):
  def __init__(self):
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])

def explore_leaves(s, v):
  ret = []
  for e in s.edges():
    s.board.push(e)
    ret.append((v(s), e))
    s.board.pop()
  return ret

if __name__ == "__main__":
  v = Valuator()
  s = State()
  while not s.board.is_game_over():
    l = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
    move = l[0]
    print(move)
    s.board.push(move[1])
  print(s.board.result())
      
      


