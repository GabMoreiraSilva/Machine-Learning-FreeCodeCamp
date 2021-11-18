# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
Markov = {}

def player(prev_play, opponent_history=[]):
  global Markov

  n = 3

  if prev_play in ["R","P","S"]:
    opponent_history.append(prev_play)

  guess = "R" # default, until statistic kicks in

  if len(opponent_history)>n:
    inp = "".join(opponent_history[-n:])

    if "".join(opponent_history[-(n+1):]) in Markov.keys():
      Markov["".join(opponent_history[-(n+1):])]+=1
    else:
      Markov["".join(opponent_history[-(n+1):])]=1

    possible =[inp+"R", inp+"P", inp+"S"]

    for i in possible:
      if not i in Markov.keys():
        Markov[i] = 0

    predict = max(possible, key=lambda key: Markov[key])

    if predict[-1] == "P":
      guess = "S"
    if predict[-1] == "R":
      guess = "P"
    if predict[-1] == "S":
      guess = "R"


  return guess