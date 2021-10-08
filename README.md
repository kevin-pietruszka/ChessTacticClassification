# Chess Tactic Classification & Provision of Conceptual Reasoning Via Machine Learning
### Background

Chess and technology have become intertwined since IBM's Deep Blue beat the best player in the world, Garry Kasparov, in 1977. Since then, chess AI’s have permanently altered how the game is played. By evaluating hundreds of thousands of possibilities, chess AI’s have enabled everyone from a novice to a grandmaster to view the game at a very analytical level. However, they struggle with giving explanations to why a move is good or bad. With the resurgence of chess into popular culture, understanding modern engines (such as Stockfish) has become a barrier to entry for chess.

### Problem Definition

These days, sites like lichess and chess.com are popular ways to play chess. These sites have built in functionalities which allow players to run an analysis of games via the chess engine Stockfish to learn more about their moves. However, Stockfish only evaluates moves numerically, which is not easy to learn from.<br><br>
We imagine a program that can analyze moves and give constructive feedback. If a player makes a bad move, our model would output something like “bad move: allows knight to be pinned on b5.” Data from puzzles, matches, and results can be evaluated to help us classify why the next move is good or bad.

Here's a more in depth example:
![chess1](https://user-images.githubusercontent.com/32807310/136492810-1b30453b-7ecd-45b8-b948-095ea8937114.JPG)<br>
In the above puzzle, stockfish recommends that Qxc3 in what appears to be a very basic trade after white responds with Rxc3:<br>
![chess2](https://user-images.githubusercontent.com/32807310/136488962-ba4af350-45b1-4261-9cac-669797859343.JPG)<br>
The tactic behind this queen trade is the followup move of Bxd4, which forks the rook on c3 and white’s king on g1:<br>
![chess3](https://user-images.githubusercontent.com/32807310/136489031-06ebc1b1-ffca-4c8f-ba52-969c1590045d.JPG)<br>

Through this three stage sequence, Stockfish has recommended all of the best possible moves, assigning them a rating corresponding to the advantage one player has over the other. In the last screenshot, Stockfish is evaluating black to have an advantage equivalent to 4.9 pawns over white, putting them in the lead. However, Stockfish fails to provide the player with any explanation/classification through these moves as to why this sequence is advantageous, which deprives the player of understanding. Our AI would recognize that Qxc3 leads to a fork and inform the player, allowing the player to quickly understand why a move is powerful without requiring prior knowledge of forking.
<br><br>

### Methods

We will use supervised learning to accomplish this task via the puzzle data sets from lichess.com. These datasets have sections dedicated to different strategies like forking with the knights that could be used to train for certain topics. In regards to the method of machine learning, a neural network or a bayes net would be used in order to learn from these data sets and attempt to classify. A bayes net approach would be a good match for the problem since the dataset contains the board states that lead up to various puzzle solutions whereas a neural network would be able to take in a wide variety of features that can be used to attempt to classify the problem. 

### Potential Results/ Discussion

Our potential result will take a given board state and give both a label (ex: fork, pin, etc.) and details relevant to that label (allows knight to be pinned on b5). As a result of this, we can let players that are trying to learn chess or trying to improve their skills improve faster, as they can be told why a certain board state is good or bad.

### Timeline + Group Member Responsibilities



Shashank Inampudi - Data Cleaner <br>
Mihir Kadiwala - Project Structurer<br>
Oliver Knauf - Data collector<br>
Aiden Melone - Data Interpreter, Deliverables guy<br>
Kevin Pietruszka - Github manager, Model Trainer<br>


![image](https://user-images.githubusercontent.com/48032258/136494906-a0c19442-9e55-457e-a400-01472a5132f5.png)


### References

Masud M., Al-Shehhi A., Al-Shamsi E., Al-Hassani S., Al-Hamoudi A., Khan L. (2015) Online Prediction of Chess Match Result. In: Cao T., Lim EP., Zhou ZH., Ho TB., Cheung D., Motoda H. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2015. Lecture Notes in Computer Science, vol 9077. Springer, Cham. https://doi.org/10.1007/978-3-319-18038-0_41

Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, and Ashton Anderson. 2020. Aligning Superhuman AI with Human Behavior: Chess as a Model System. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 1677–1687. DOI:https://doi.org/10.1145/3394486.3403219

Reiser P.G.K., Riddle P.J. (1999) Evolving Logic Programs to Classify Chess-Endgame Positions. In: McKay B., Yao X., Newton C.S., Kim JH., Furuhashi T. (eds) Simulated Evolution and Learning. SEAL 1998. Lecture Notes in Computer Science, vol 1585. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-48873-1_19
