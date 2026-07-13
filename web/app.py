from flask import Flask, request, jsonify, session, render_template

from agents import get_action
from ConnectFour.connect_four_env import ConnectFourEnvironment

app = Flask(__name__)
app.secret_key = "dev"

def _c4_restore():
    board = session.get("board")
    current_player = session.get("current_player", 1)
    fresh_env = ConnectFourEnvironment()
    fresh_env.board = board
    fresh_env.current_player = current_player
    return fresh_env

def _c4_serialize(env):
    return {
        "board": env.board,
        "current_player": env.current_player,
    }

def _c4_display(env):
    return [[cell * env.current_player for cell in row] for row in env.board]

