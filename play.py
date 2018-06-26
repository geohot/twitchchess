#!/usr/bin/env python3
import chess
import time
import torch
import chess.svg
import traceback
import base64
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

# chess board and "engine"
s = State()
v = Valuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
app = Flask(__name__)

@app.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())

def computer_move(s, v):
  # computer move
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  print("top 3:")
  for i,m in enumerate(move[0:3]):
    print("  ",m)
  s.board.push(move[0][1])

@app.route("/selfplay")
def selfplay():
  s = State()

  ret = '<html><head>'
  # self play
  while not s.board.is_game_over():
    computer_move(s, v)
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(s)
  print(s.board.result())

  return ret


@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get('move',default="")
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
  else:
    print("GAME IS OVER")
    response = app.response_class(
      response="game over",
      status=200
    )
    return response
  print("hello ran")
  return hello()

if __name__ == "__main__":
  app.run(debug=True)

