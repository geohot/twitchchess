#!/usr/bin/env python3
#from __future__ import print_function
import os
import chess
import time
from datetime import datetime
import chess.svg
import chess.pgn
import traceback
import base64
import random
import sys
from state import State
from colorama import Fore, Back, Style, init
import termcolor
init(autoreset=True)
from termcolor import colored
from halo import Halo
import pickle
import numpy as np
import pandas as pd

class Valuator(object):
  def __init__(self):
    import torch
    from train import Net
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])

# let's write a simple chess value function
# discussing with friends how simple a minimax + value function can beat me
# i'm rated about a 1500

MAXVAL = 10000
class ClassicValuator(object):
  values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

  def __init__(self):
    self.reset()
    self.memo = {}

  def reset(self):
    self.count = 0

  # writing a simple value function based on pieces
  # good ideas:
  # https://en.wikipedia.org/wiki/Evaluation_function#In_chess
  def __call__(self, s):
    self.count += 1
    key = s.key()
    if key not in self.memo:
      self.memo[key] = self.value(s)
    return self.memo[key]

  def value(self, s):
    b = s.board
    # game over values
    if b.is_game_over():
      if b.result() == "1-0":
        return MAXVAL
      elif b.result() == "0-1":
        return -MAXVAL
      else:
        return 0

    val = 0.0
    # piece values
    pm = s.board.piece_map()
    for x in pm:
      tval = self.values[pm[x].piece_type]
      if pm[x].color == chess.WHITE:
        val += tval
      else:
        val -= tval

    # add a number of legal moves term
    bak = b.turn
    b.turn = chess.WHITE
    val += 0.1 * b.legal_moves.count()
    b.turn = chess.BLACK
    val -= 0.1 * b.legal_moves.count()
    b.turn = bak

    return val

def computer_minimax(s, v, depth, a, b, big=False):
  
  if depth >= 5 or s.board.is_game_over():
    return v(s)
  # white is maximizing player
  turn = s.board.turn
  if turn == chess.WHITE:
    ret = -MAXVAL
  else:
    ret = MAXVAL
  if big:
    bret = []
    
  # can prune here with beam search
  isort = []
  for e in s.board.legal_moves:
    s.board.push(e)
    isort.append((v(s), e))
    s.board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

  # beam search beyond depth 3
  if depth >= 3:
    move = move[:10]
  for e in [x[1] for x in move]:
    s.board.push(e)
    
    tval = computer_minimax(s, v, depth+1, a, b)
    s.board.pop()
    if big:
      bret.append((tval, e))
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b:
        break  # b cut-off
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b:
        break  # a cut-off
  
  if big:
    return ret, bret
  else:
    return ret
def explore_leaves(s, v):
  ret = []
  rndm = random.randint(2,12)
  print(colored(Style.BRIGHT + 'Thinking about your move ','magenta') + colored(str(rndm),'cyan') + colored(Style.BRIGHT + ' Out.... Probably more O.o','magenta'))
  spinner = Halo(text='SEARCHING MINDSTATE',text_color='cyan', spinner='simpleDotsScrolling',color='cyan')
  spinner.start()
  start = time.time()
  v.reset()
  bval = v(s)
  try:
    cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
  except:
    cval = 0 
    ret = []
  eta = time.time() - start
  spinner.stop()
  print(colored(Style.BRIGHT + "%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec",'yellow') % (bval, cval, v.count, eta, int(v.count/eta)))
  return ret

# chess board and "engine"
s = State()
#v = Valuator()
v = ClassicValuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

from flask import Flask, Response, request
application = Flask(__name__)

winners = np.array([])

@application.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())

def computer_move(s, v):
  # computer move
  try:
    move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  except:
    move = []
  if len(move) == 0:
    m1 = "game over"
    return m1
  print(colored(Style.BRIGHT + "top 3:",'green'))
  for i,m in enumerate(move[0:3]):
    mi = str(m)
    m1 = mi.split("(",)[1]
    m2 = m1.split(",",)[0]
    m= mi.split("'",)[1]
    print("  ",colored(Style.DIM+ "Value increase: ",'green'),colored(Style.BRIGHT + m2,'cyan'),colored(Style.DIM + " for move ",'green'),colored(Style.BRIGHT + m,'cyan'))
    if s.board.turn == False:
        comp = colored(Back.WHITE + Fore.BLACK + Style.DIM + "Agent-K")
    else:
        comp = colored(Back.MAGENTA + Fore.CYAN + Style.BRIGHT + "Agent-J")
  #readout = str(comp, colored(Style.BRIGHT + "moving",'magenta'), colored(Style.BRIGHT + str(move[0][1]),'yellow'))
  print(comp, colored(Style.BRIGHT + "moving",'magenta'), colored(Style.BRIGHT + str(move[0][1]),'yellow'))
  s.board.push(move[0][1])
  m1 = str(move[0][1])
  #game = chess.pgn.Game()
  return m1,s

i = 0
with open('rnd.pickle','wb') as rnd:
        pickle.dump(i,rnd)
moves = []
with open('g.pickle','wb') as g:
    pickle.dump(moves, g)
@application.route("/selfplay")
def selfplay():
    m = request.args.get('m', default='')
    with open('rnd.pickle','rb') as rnd:
        i = pickle.load(rnd)
        i = 1 + i
    with open('rnd.pickle','wb') as rnd:
        pickle.dump(i,rnd)
    if m == '':
        #print('got1')
        s = State()          
        #s = request.args.get('s', default='')
        ret = 'game over'
        while not s.board.is_game_over():
            print(colored('move: ' + str(i),'yellow'))
            print(colored(m,'cyan'))
            m2,si = computer_move(s, v)
            with open('si.pickle', 'wb') as p:
                pickle.dump(si, p)
            with open('g.pickle','rb')as g:
                moves = pickle.load(g)
                moves.append(m2)
            with open('g.pickle','wb')as g:
                pickle.dump(moves, g)
                print(colored(moves,'cyan'))
            m1 = m2 + ":"
            response = application.response_class(
            response=m1 + s.board.fen(),
              status=200
            )
            return response
        return ret
    else:
        ret = 'game over'
        with open('si.pickle', 'rb') as f:
                si = pickle.load(f)
        while not si.board.is_game_over():
            m2,sii = computer_move(si, v)
            with open('si.pickle', 'wb') as p:
                pickle.dump(sii, p)
            print(colored('move: ' + str(i),'yellow'))
            #ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(sii)
            with open('g.pickle','rb')as g:
                moves = pickle.load(g)
                moves.append(m2)
            with open('g.pickle','wb')as g:
                pickle.dump(moves, g)
                print(colored(moves,'cyan'))
            m1 = m2 + ":"
            response = application.response_class(
            response=m1 + si.board.fen(),
                status=200
            )
            return response
        return ret

# move given in algebraic notation
@application.route("/move")
def move():
  if not s.board.is_game_over():
    move = s.board.san(chess.Move(move[0][1]))
    if move is not None and move != "":
      print(colored(Style.BRIGHT + "human moves",'cyan'), colored(Style.BRIGHT + move,'green'))
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
    response = application.response_class(
      response=s.board.fen(),
      status=200
    )
    return response
  else:
    print(colored(Style.BRIGHT + "********************* GAME IS OVER *********************",'red'))
    response = application.response_class(
      response="game over",
      status=200
    )
    return response
  print("hello ran")
  return hello()

# moves given as coordinates of piece moved
@application.route("/move_coordinates")
def move_coordinates():
  
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    sauce = request.args.get('sauce', default='')
    target = int(request.args.get('to', default=''))
    targe = request.args.get('targe', default='')
    promotion = True if request.args.get('promotion', default='') == 'true' else False
    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    if move is not None and move != "":
      print(colored(Style.BRIGHT + "human moves",'cyan'), colored(Style.BRIGHT + move,'green'))
      
      m1 = ':'
      try:
        s.board.push_san(move)
        with open('g.pickle','rb')as g:
                moves = pickle.load(g)
                moves = moves.append(move)
        with open('g.pickle','wb')as g:
                pickle.dump(moves, g)
                print(colored(moves,'cyan'))
        m2,s1 = computer_move(s, v)
        with open('g.pickle','rb')as g:
                moves = pickle.load(g)
                moves = moves.append(m2)
        with open('g.pickle','wb')as g:
                pickle.dump(moves, g)
                print(colored(moves,'cyan'))
        m1 = m2 + ":"
      except Exception:
        traceback.print_exc()
      response = application.response_class(
      response=m1 + s.board.fen(),
      status=200
      )
    return response
  print(colored(Style.BRIGHT + "********************* GAME IS OVER *********************",'red'))
  response = application.response_class(
    response="game over",
    status=200
  )
  return response



@application.route("/post")
def post():
  game = chess.pgn.Game()
  with open('si.pickle', 'rb') as f:
    si = pickle.load(f)
    b = si.board
    # game over values
    if b.is_game_over():
      if b.result() == "1-0":
        game.headers["Result"] = "1-0"
        winners = np.append(1)
      elif b.result() == "0-1":
        game.headers["Result"] = "0-1"
        winners = np.append(0)
      else: 
        winners = np.append(2)
  win = open('winner.txt', 'a')
  sgame = str(game)
  win.write(winner + sgame)
  game.headers["Event"] = "Agentjk"
  game.headers["Site"] = "local"
  game.headers["Date"] = datetime.now()
  game.headers["White"] = "Agent-J"
  game.headers["Black"] = "Agent-K"
  game.end()
  print(game)
  with open('g.pickle','rb')as g:
        moves = pickle.load(g)
        print(colored("Final Game",'cyan') + moves)
        node = game.add_variations(chess.Move.from_uci(moves[0]))
        for m in moves:
            node = node.add_variations(chess.Move.from_uci(m))
        log = open('/TrainingGames/'+ str(datetime.now())+".pgn", 'a')
        log.write(game)
  Col = ['Winner']
  df = pd.DataFrame(winners,columns=Col)
  html = df.to_html()
  response = application.response_class(
    response=html,
    status=200
  )
  return response

@application.route("/newgame")
def newgame():
  s.board.reset()
  response = application.response_class(
    response=s.board.fen(),
    status=200
  )
  return response

@application.route("/undo")
def undo():
  s.board.pop()
  response = application.response_class(
    response=s.board.fen(),
    status=200
  )
  return response

if __name__ == "__main__":
  if os.getenv("SELFPLAY") is not None:
        s = State()
        print(s.board)
        print(s.board.result())
  else:
        application.run(debug=True)
        
