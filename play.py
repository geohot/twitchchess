#!/usr/bin/env python3
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

if __name__ == "__main__":
  v = Valuator()
  s = State()
  print(v(s))
  for e in s.edges():
    s.board.push(e)
    print(e, v(s))
    s.board.pop()

