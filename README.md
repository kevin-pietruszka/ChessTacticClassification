# CS4641
Background
Chess and technology have become inextricably intertwined since the first AI, IBM's Deep Blue, beat the best player in the world, Garry Kasparov, in 1977. Since then, chess AI’s have permanently altered how the game is played. By being able to evaluate hundreds of thousands of possible realities in an instant, chess AI’s have enabled everyone from a novice to a grandmaster to view the game at a very technical, analytical level. However, they struggle with communicating conceptual ideas and giving explanations to why a move is good or bad. And, with the resurgence of chess into popular culture, understanding modern engines (such as StockFish) has become a barrier to entry in learning chess.

Problem Definition
Online chess players use features on sites like chess.com or Lichess to review their games everyday, but they do not give information on why a certain move is good or bad. For those that are looking to get better at playing chess, the puzzles or online matches on Lichess can be a great tool for improvement. However, Stockfish simply evaluates the move on a scale that does not provide sufficient, understandable feedback to the user to help them better their skills.
We imagine a program that can analyze moves and give constructive feedback that can be helpful to an intermediate chess player. If they make a bad move that could get a piece stuck, it would output something like “bad move: allows knight to be pinned on b5.” Data from puzzles, matches, and results can be evaluated to help us classify why the next move given by Stockfish is a good move.

Methods
We will use supervised learning in order to accomplish this task by using the puzzle data sets from lichess.com. These data sets have sections dedicated to different strategies like forking with the knights that could be used to train for certain topics. In regards to the method of machine learning, a neural network or a bayes net would be used in order to learn from these data sets and attempt to classify. A bayes net approach would be a good match for the problem since the dataset contains the board states that lead up to various puzzle solutions whereas a neural network would be able to take in a wide variety of features that can be used to attempt to classify the problem. 

Potential Results/ Discussion
Our potential result will take a given board state and give both a label (ex: fork, pin, etc.) and details relevant to that label (allows knight to be pinned on b5). As a result of this, we can let players that are trying to learn chess or trying to improve their skills improve faster, as they can be told why a certain board state is good or bad.

Timeline + Group Member Responsibilities



Shashank Inampudi - Data Cleaner
Mihir Kadiwala - Project Structurer
Oliver Knauf - Data collector
Aiden Melone - Data Interpreter, Deliverables guy
Kevin Pietruszka - Github manager, Model Trainer




References
Masud M., Al-Shehhi A., Al-Shamsi E., Al-Hassani S., Al-Hamoudi A., Khan L. (2015) Online Prediction of Chess Match Result. In: Cao T., Lim EP., Zhou ZH., Ho TB., Cheung D., Motoda H. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2015. Lecture Notes in Computer Science, vol 9077. Springer, Cham. https://doi.org/10.1007/978-3-319-18038-0_41

Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg, and Ashton Anderson. 2020. Aligning Superhuman AI with Human Behavior: Chess as a Model System. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 1677–1687. DOI:https://doi.org/10.1145/3394486.3403219

Reiser P.G.K., Riddle P.J. (1999) Evolving Logic Programs to Classify Chess-Endgame Positions. In: McKay B., Yao X., Newton C.S., Kim JH., Furuhashi T. (eds) Simulated Evolution and Learning. SEAL 1998. Lecture Notes in Computer Science, vol 1585. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-48873-1_19
