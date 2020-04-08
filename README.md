# twitchchess

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/m1.PNG" />
Agents K and J will play selves
Agent K will play white = human

modded from git GEOHOT/twitchchess
- Added more HTML feeback and colorized console, added more exceptions and feedback in waiting.
- trained on grandmaster games to .017% loss rate saved to new net
- built in replies from index.html and a game posting function
- built log for saving to TrainingGames/ and post()
	- Randomized first move in selfplay for training sets. commented out original
- debugged UI a bit

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/m2.PNG" />

Its still a bit buggy interface but: 
- moving a peice off board will reset the peice to original position
- to manually post a game to pgn use concede and post button below board
- must reset board with new game befor next human game
- selfplay will do its thing for you

<img width=600px src="https://raw.githubusercontent.com/Encryptic1/twitchchess/master/hm1.PNG" />

human play will give both shell and html feedack
- while the program is exploing leaves any input will cause a break
- must wait until your turn to ff 
- new game resets board without posting game

	TODO:
	- selftrain set
	-  add policy condition(make utility move) for tie values, in lategame the bot plays safe. wheres the aggro?
	

