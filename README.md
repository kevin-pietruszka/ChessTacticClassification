# Chess AI Tactic Classification & Provision of Conceptual Reasoning.
### Background

Chess and technology have become inextricably intertwined since the first AI, IBM's Deep Blue, beat the best player in the world, Garry Kasparov, in 1977. Since then, chess AI’s have permanently altered how the game is played. By being able to evaluate hundreds of thousands of possible realities in an instant, chess AI’s have enabled everyone from a novice to a grandmaster to view the game at a very technical, analytical level. However, they struggle with communicating conceptual ideas and giving explanations to why a move is good or bad. And, with the resurgence of chess into popular culture, understanding modern engines (such as Stockfish) has become a barrier to entry in learning chess.

### Problem Definition

These days, sites like lichess and chess.com have become the most popular ways to play chess. These sites have built in functionalities which allow players to run an analytical analysis of games via usage of the chess engine Stockfish in the hopes of learning more about the powerful and weak plays they made. However, Stockfish only evaluates moves on a purely numerical scale which does not present the user with a particulary friendly opprotunity to learn.<br><br>
We imagine a program that can analyze moves and give constructive feedback which can be helpful to an intermediate chess player. If they make a bad move that could get a piece stuck, it would output something like “bad move: allows knight to be pinned on b5.” Data from puzzles, matches, and results can be evaluated to help us classify why the next move given by Stockfish is a good move.

Here's a more in depth example:
![chess1](https://user-images.githubusercontent.com/32807310/136488491-de3785d9-a4e9-4241-bd48-ea586e24ecb1.png)
In the above puzzle, stockfish recommends that Qxc3 in what appears to be a very basic trade after white responds with Rxc3:
![chess2](https://user-images.githubusercontent.com/32807310/136488962-ba4af350-45b1-4261-9cac-669797859343.JPG)
The tactic behind this queen trade is the followup move of Bxd4, which forks the rook on c3 and white’s king on g1:
![chess3](https://user-images.githubusercontent.com/32807310/136489031-06ebc1b1-ffca-4c8f-ba52-969c1590045d.JPG)

Through this three stage sequence, Stockfish has recommended all of the best possible moves, specifically assigning them a rating which corresponds to the advantage, in pawns, one player has over the other. In the last screenshot, Stockfish is evaluating black to have an advantage equivalent to 4.9 pawns over white, putting them handedly in the lead. However, Stockfish fails to provide the player with any explanation/classification through these steps of moves as to why this sequence is so advantageous, which deprives the player of a learning opportunity. Our AI would immediately recognize that Qxc3 leads to a fork and inform the player of this, which would allow the player to reach a better understanding why a move is powerful quicker and without requiring prior knowledge to the concept of a fork.
<br><br>
Of course, this functionality would become more and more useful as situations increase in complexity, especially in scenarios in which a player made a poor move where the AI could point out what the player left themselves open to, even if that situation never played out in the game being analyzed.



### Methods

We will use supervised learning in order to accomplish this task by using the puzzle data sets from lichess.com. These data sets have sections dedicated to different strategies like forking with the knights that could be used to train for certain topics. In regards to the method of machine learning, a neural network or a bayes net would be used in order to learn from these data sets and attempt to classify. A bayes net approach would be a good match for the problem since the dataset contains the board states that lead up to various puzzle solutions whereas a neural network would be able to take in a wide variety of features that can be used to attempt to classify the problem. 

### Potential Results/ Discussion

Our potential result will take a given board state and give both a label (ex: fork, pin, etc.) and details relevant to that label (allows knight to be pinned on b5). As a result of this, we can let players that are trying to learn chess or trying to improve their skills improve faster, as they can be told why a certain board state is good or bad.

### Timeline + Group Member Responsibilities



Shashank Inampudi - Data Cleaner <br>
Mihir Kadiwala - Project Structurer<br>
Oliver Knauf - Data collector<br>
Aiden Melone - Data Interpreter, Deliverables guy<br>
Kevin Pietruszka - Github manager, Model Trainer<br>




### References

Masud M., Al-Shehhi A., Al-Shamsi E., Al-Hassani S., Al-Hamoudi A., Khan L. (2015) Online Prediction of Chess Match Result. In: Cao T., Lim EP., Zhou ZH., Ho TB., Cheung D., Motoda H. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2015. Lecture Notes in Computer Science, vol 9077. Springer, Cham. https://doi.org/10.1007/978-3-319-18038-0_41

Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, and Ashton Anderson. 2020. Aligning Superhuman AI with Human Behavior: Chess as a Model System. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 1677–1687. DOI:https://doi.org/10.1145/3394486.3403219

Reiser P.G.K., Riddle P.J. (1999) Evolving Logic Programs to Classify Chess-Endgame Positions. In: McKay B., Yao X., Newton C.S., Kim JH., Furuhashi T. (eds) Simulated Evolution and Learning. SEAL 1998. Lecture Notes in Computer Science, vol 1585. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-48873-1_19
