# Chess Tactic Classification & Provision of Conceptual Reasoning Via Machine Learning

**Background**

Chess and technology have become intertwined since IBM&#39;s Deep Blue beat the best player in the world, Garry Kasparov, in 1977. Since then, chess AI&#39;s have permanently altered how the game is played. By evaluating hundreds of thousands of possibilities, chess AI&#39;s have enabled everyone from a novice to a grandmaster to view the game at a very analytical level. However, they struggle with giving explanations to why a move is good or bad. With the resurgence of chess into popular culture, understanding modern engines (such as Stockfish) has become a barrier to entry for chess.

**Problem Definition**

Online chess players use chess engines on sites like chess.com or Lichess to review their games everyday, but they are not given information on why certain moves are good or bad. For students of chess, the puzzles on these sites can present opportunities for these more conceptual ideas to be presented and learned. However, during game analysis, engines like Stockfish simply evaluate moves on a numerical scale which lacks the conceptual feedback a user can directly apply to improve their skills.

We imagine a program that can analyze moves and give helpful constructive feedback to intermediate chess players. If a user made a bad move that could get a piece stuck, it would output something like &quot;bad move: allows knight to be pinned on b5.&quot; Data from puzzles, matches, and results can be evaluated to help us classify why the next move given by Stockfish is a good move.

Example:

In the following puzzle, stockfish recommends that Qxc3 in what appears to be a very basic trade after white responds with Rxc3: 

![chess1](https://user-images.githubusercontent.com/70602902/145196361-832befcd-16a9-43e3-b4eb-f00aebaf02cb.png)

The tactic behind this queen trade is the followup move of Bxd4, which forks the rook on c3 and white&#39;s king on g1: 

![chess2](https://user-images.githubusercontent.com/70602902/145196363-bc4e1ea8-7d50-4eaa-84f0-154a51cfda29.png)

This sequence is relatively easy to understand after tracing to its conclusion. However, as situations become increasingly complicated/ambiguous, it would be useful to know what a certain move insinuates, especially in more complex scenarios. Our AI can provide players with conceptual explanations as to WHY a move is impactful. In the above example, our algorithm will still recommend Qxc3 but would also define that the exchange would lead to a fork.

**Data Collection**

Our raw data was sourced from [https://database.lichess.org/#puzzles](https://database.lichess.org/#puzzles), which contains a csv hosting all of the puzzles hosted and collected on lichess.com. This looked like the following: 

![data](https://user-images.githubusercontent.com/70602902/145196365-7e50968d-cc7e-447a-8cdf-505a6da769c2.png)

Each entry in the CSV contained both the initial board state in FEN notation (column 2), the moves necessary to finish the puzzle (column 3) and &quot;theme&quot; tags (column 8) categorizing which type of puzzle each instance was. We then cleaned the data, which happened in a few steps.

The first step, we eliminated all puzzles that did not correspond to a theme of interest. In particular, we considered themes which were of both high numerical count and of low ambiguity. For example, themes like &quot;crushing&quot; or &quot;bishop endgame&quot; were cases that we believed to be too ambiguous for our model to be able to properly define, at least based on the amount of data available.

After doing this, we deleted the columns containing information irrelevant to our use case. Information such as puzzle ELO rating, lichess link, and lichess label were deleted until we were left with only the starting chess puzzle position, moves necessary to complete the puzzle, and cleaned themes.

It proved to be essential to consider the range in puzzle complexity the dataset presented. Simple puzzles could be solved in just two moves, while the more complex ones might take 8+ steps. Since some puzzles in our dataset took varying numbers of moves to solve, we organized the data by the different number of moves that are required to solve a puzzle. Such organization revealed that the more extreme complexity (10+ move) cases lack a significant amount of data (\&lt;21,000) given the abstract nature of our problem. This led to us only considering datasets for the puzzles that are of length 2, 4, 6, and 8 moves to solve in different files.

Additionally, these datasets were used to generate more features utilized by the decision tree such as the stockfish evaluation of the various positions, the material gained, the piece that was moved the most, and if a checkmate was involved. All of these features could be calculated parsing the FEN from the datasets, however, as the features got more and more complicated the time complexity of collecting this data increased dramatically.

**Data Cleaning Technologies &amp; Techniques**

We utilized the libraries of Pandas, Numpy, and python-chess to prepare our data for our models. Python-chess was the library that we used to take the standard chess notation from our data and organize it into something that Tensorflow, our library used for machine learning algorithms, could better understand. This meant that we did not need to manually change the string notations into numbers for our very large dataset, and instead, we simply utilized a library to do it for us. To begin our model, we first needed to organize the cleaned data into matrices that could be readable to a supervised learning algorithm. We used methods common to other machine learning analysis of chess data and split our information into many 12 8x8 matrices per board position to create a large multidimensional array which contained the position of each piece on the board. In more complex detail, these 8x8 arrays were one hot encoded to each of black and white&#39;s chess pieces. For example, the first 8x8 array was one hot encoded to display the location of white&#39;s pawns, the next array white&#39;s knights, etc. and, starting at the 7th array, the process was repeated for black.

Likewise, In order to better obtain our results, we needed to change the way that we thought about our labels. This is because in some other machine learning problems labels can be used to wholly represent a specific piece of data, but in our case, we are dealing with the possibilities of multiple classifications per piece of data. The issue with the ordinary approach of giving one data point one label is that a specific data can actually belong to multiple labels. We have labels like fork, pin, and skewer. This is all chess jargon for different scenarios that can occur with the pieces, but multiple scenarios can occur at once. It is possible to both pin and fork your opponent at the same time. To accommodate for this, we used the one-hot technique. We used an array of the same size as the number of our labels, and placed a 1 in the corresponding column if the data point was assigned that label. So if we had 6 labels and a certain puzzle was assigned fork and pin, then the one-hot representation of this would be [0, 1, 0, 1, 0, 0] if the 1st and 3rd index indicated fork and pin.

Furthermore, we also split the cleaned data into training and testing groups. We decided on using 80% of our data for training and the other 20% for testing. We are aware of the fact that many other models may have higher proportions of their dataset used for training, but we believe that 80% would be a good margin given the magnitude of the dataset and our concerns of overfitting.

**Considered Methodologies and Results**

**Neural Network Phase 1**

Tensorflow&#39;s Keras neural network implementation was initially deemed to be the most fitting supervised learning technique we could implement. This was largely because of its modularity, wealth of online documentation, ease of use, and ability to handle multi classification scenarios such as our own. This iteration of the neural network was purposefully basic and served as an adequate proof of concept. The model took advantage of the tf.keras.Sequential() library, which makes sense for our use case as we are inputting one large tensor corresponding to the shape/dimensions of #puzzles x #number of moves x (12 x 8 x 8 ), which was the board representation previously discussed. To help the model process this information, there is a flatten layer which transforms the data into a more parsable format. Following this layer, there are two intermediary dense layers with 128 and 64 nodes respectively, both with activation function &quot;relu&quot;, which we discovered to be the best for our use case after some research and reference to previous projects. Finally, we have an output layer of 6 nodes, corresponding to the 6 puzzle theme types &quot;hanging piece&quot;, &quot;fork&quot;, &quot;trapped piece&quot;, &quot;pin&quot;, &quot;backrank mate&quot;, and &quot;skewer&quot; we used as a test case with an activation function of softmax to account for multi classification possibilities.

For assessing our model, we chose to use the Adam optimizer, a loss function of categorical\_crossentropy, the default learning rate of .001, an output metric of accuracy, and an epoch count of 10. Based on our research, both the Adam optimizer and categorical\_crossentropy have seen wide use in one hot and multi classification examples, so we followed in those footsteps for this proof of concept, but would go on to experiment with other potentials in the future. Of course, accuracy was also output to define an easily conceptualized metric that evaluated the overall success of our algorithm. Finally, we chose to use the relatively low epoch count of 10 as we were still fine tuning our model and for time&#39;s sake wished to keep training short and additionally saw only a negligible difference in accuracy over time past 10 epochs.

**Neural Network Phase 1 Results**

Recall that we split our data into the different number of moves that are required to solve the puzzles. We ran our algorithm for 6 of these datasets: 2, 4, 6, 8, and 10. Because of the way we captured our data, we simply needed to change the parameter in our call to designModel() in order to reference another dataset and run the model. Below, we have the results of the algorithm for the puzzles that required two moves to solve.

![2 loss](https://user-images.githubusercontent.com/70602902/145196351-0113e645-389d-4c56-bd14-a27d647ee8c4.png)

![2 acc](https://user-images.githubusercontent.com/70602902/145196349-9aebe637-bc4e-469f-9936-fb705ec194eb.png)

Testing accuracy: .708

Test Loss: 1981.205

Here is our model for puzzles with 4 moves:

![4 loss](https://user-images.githubusercontent.com/70602902/145196353-fceeca96-df34-4d9c-aaa1-b95c4c29a352.png)

![4 acc](https://user-images.githubusercontent.com/70602902/145196352-b8297d73-1362-43c6-a0e5-c648f82c8b0d.png)

Testing accuracy: .479

Test Loss: 240209.563

Now here are our graphs for the rest of the datasets.

![6 loss](https://user-images.githubusercontent.com/70602902/145196356-358c601f-96c5-480d-96e9-868314bd29ad.png)

![6 acc](https://user-images.githubusercontent.com/70602902/145196354-948c8b46-1f97-4f2b-8817-f2756fec6005.png)

Testing accuracy: .49

Test Loss: 77623.359

![8 loss](https://user-images.githubusercontent.com/70602902/145196358-1ca0112f-7c57-4a91-b3b6-369fd6d7c11f.png)

![8 acc](https://user-images.githubusercontent.com/70602902/145196357-d2a3f443-43c4-4dd4-bf3c-57c82dce73c6.png)

Testing accuracy: .530

Test Loss: 7406.889

![10 loss](https://user-images.githubusercontent.com/70602902/145196708-c93abec4-bba1-4165-bb39-3fd85f5a7c47.png)

![10 acc](https://user-images.githubusercontent.com/70602902/145196705-d11e5f60-d71e-441f-a276-3f3bbaf31191.png)

Testing accuracy: .512

Test Loss: 1023.130

**Neural Network Phase 1 Discussion**

We used a NN model because puzzles and the game of chess greatly depend on the previous states of the board.

In terms of numerical relationships, models which had a higher volume of input data expectedly corresponded to high loss values; model 4, which had the most training data (~440,000) had the most loss (240209.563) whereas model 10, which had the least training data (~7,800) had the least loss (1023.130). Model accuracy was more complexity correlated, seemingly depending on a combination of the amount of training data and the number of puzzles/moves/input features. The model had a .703 accuracy for model 2 and was more successful by comparison to the rest of the models.

One thing to take away from the results is that although the accuracy is much higher than a guess, the model still makes many mistakes. This means that if this were to eventually be displayed as a tool for the online chess community the accuracy needs to be higher, which has led us to look for ways to improve the results. Since we chose to split the data into the different moves required for the puzzle, we had a smaller amount of data than before. The smaller dataset had a negative impact on the accuracy, but this was necessary because of the large variance between the types of puzzles.

We have tuned our parameters and used multiple methods for conducting our neural network including stochastic gradient descent, Adam&#39;s optimizer, and root mean squared propagation. Each of these had their own tuning necessary and we were able to find the best ones that worked for us after much trial and error.

**Decision Tree**

After doing some experimenting with sklearn&#39;s library for decision trees, it was decided to see if Decision Trees could lead to different results. As mentioned in the data collection section, using the puzzle data that was collected, the information could be used to calculate various pieces of information that could be used as features for a decision tree. The features selected were good for our small choice of desired themes since many of the features could provide distinct information about the labels, but they were extensive to calculate and greatly increased the time complexity of collecting the data required. From here all that was needed was to slice the data into training sets and test sets where accuracy was measured by the number of labels that were not the same.

**Decision Tree Results &amp; Discussion**

![tree](https://user-images.githubusercontent.com/70602902/145196367-f2832443-4d92-4375-9231-7687e632c0e5.png)

The decision tree, overall, kept a steady accuracy at around 60% for most of the datasets. Although this value seems impressive, these sets only included 6 desired themes out of countless others that were simpler to classify. To expand this model, to include more themes, more features would have to be included in order to help the decision tree obtain more information for its branches. This causes two problems. For one, generating more features for the data would mean that manual pattern recognition would have to be put into place in order to determine features that would be relevant to the new features. Second, increasing the amount of themes and likewise features, greatly increases the time complexity of generating the data and training the model. To develop the current features used, stockfish&#39;s algorithms have to be run on each FEN which greatly increases the time to parse all of the raw data. Due to these reasons, a decision tree, although showing promising results, would ultimately not be feasible for future iterations and expansions of the project.

**Neural Network Phase 2**

Reflection upon the decaying accuracy and exponentially increasing loss trends displayed in the neural network phase 1 demonstrated that our model was experiencing a severe case of overfitting. With this idea in mind, we set out to rebuild our model from the ground up with the capability of classifying 3 times as many themes (6 -\&gt; 18). This goal was achieved by taking the original neural network and adding in dropout layers with a rate of .25 after each Dense layer in addition to an entirely new Dense layer before the output layer of node amount 32. The number of nodes in the output layer was also increased to match the new number of themes to be classified (18). Furthermore, deeper research into previous multi-classification neural networks with tensorflow revealed that, along with Adam, RMSprop and SGD are two other approaches commonly implemented. This led our team to run our new neural network with each of these optimizers for each number of puzzle moves in order to determine which optimizers worked best when. Notably, the optimizer learning rate was decreased by a factor of 100 to .00001 in an effort to cut down on potential overfitting. The final change made to the neural network was a doubling in epochs, which allowed for trends in accuracy and loss to make themselves more evident over time.

**Neural Network Phase 2 Results**

2 Moves: ![2 Moves](https://user-images.githubusercontent.com/70602902/145195609-89c10621-215f-428e-aead-d513d9b747cf.png)

4 Moves: ![4 Moves](https://user-images.githubusercontent.com/70602902/145195610-bc0b3cf6-2aaf-400f-954f-69f15af1d757.png)

6 Moves: ![6 Moves](https://user-images.githubusercontent.com/70602902/145195611-bd71874d-a9fe-4f5b-a79c-a085a02bd77b.png)

8 Moves: ![8 Moves](https://user-images.githubusercontent.com/70602902/145195612-7c3ad394-c345-407e-84bf-ac2aa5956197.png)

Overall Testing Accuracy: 

![acc](https://user-images.githubusercontent.com/70602902/145196360-036d9a11-aa93-4c86-8777-cee9c06450e7.png)

Overall Testing Loss (log scale): 

![acc log](https://user-images.githubusercontent.com/70602902/145196359-8ad25517-93ab-4d29-834f-44da086d8f03.png)

**Neural Network Phase 2 Discussion**

The first case tested in the neural network phase two implementation was the two move puzzle condition, and, immediately, we noticed improvements in trends and efficiencies. Whereas the first neural network implementation showed a drop off in accuracy and exponential increase in loss over epochs, our new model&#39;s anti-overfitting measures solved for these problems and introduced more positive trends into our data. At least one optimizer function would present a positive trend in each of the puzzle length cases tested, and, in the case of two move puzzles, all optimizers showed hopeful trends, with Adam and RMSprop conveying the highest level of testing accuracy at ~69%. Notably, however, SDG&#39;s accuracy trend seemed to be nearly exponential, conveying that, perhaps with an increase in epochs, it could surpass the other two optimizers.

The four move puzzle case displayed an unfortunate relationship between accuracy and loss for both Adam and RMSprop over epochs, with SGD being the only optimizer able to keep loss under control and flatten off accuracy at ~27%. A possible explanation for the poor performance of Adam and RMSprop in this case is related to the fact that the four move puzzle test case dataset is the largest of the considered by a wide margin at ~809,000 entries. By comparison, the two move case contained ~60,000 entries. Such a massive increase in entries in combination with a minimal increase in features- two more board states- likely caused these optimizers to fall prey to overfitting, implying that they would have benefited from an increase in drop layer rates and/or decrease in learning rate.

The six move puzzle case shows that the difference in loss between Adam/RMSprop and SGD became even more severe with epochs. It additionally, interestingly, conveys that, despite the massive amount of loss, Adam had the highest amount of accuracy at ~25%. Similar trends were seen in the first implementation of a neural network for this scenario, which, in other cases, have had their trends improved by targeting overfitting, implying that overfitting may be the culprit in this situation as well.

Finally, the eight move puzzle shows some return to form in magnitude of loss for Adam and RMSprop optimizers, implying that, as the number of puzzles begin to drop once again, those optimizers begin to handle the scenarios better. However, it is important to note that both Adam and RMSprop are displaying exponential increases in loss over time and a decay in accuracy. This stands in contrast with SGD which shows a linear increase in accuracy and linear decrease in loss over epochs. So, while all three of these optimizers may have converged upon the same accuracy of ~13%, given more epochs, it is likely that SGD would have outperformed the other optimizers.

Overall, the best case accuracy decreased over time between the three optimizers moving from ~69% -\&gt; ~27% -\&gt; ~25% -\&gt; ~13%. This matches up with our team&#39;s hypothesis, which is that as puzzle depth/complexity increases, our neural network&#39;s ability to assess the data given decreases. Additionally, the more severe decreases in accuracy, such as ~69% -\&gt; ~27% can be explained by what is likely overfitting. As such, if this study were to be repeated, dropout/learning rate for the 4 move puzzle condition onward should be adjusted as previously discussed.

**Conclusion**

To recap on the content covered within this paper, the original motivation was to build a technology which had the capability of providing new chess players with an answer to &quot;WHY?&quot; a move recommended by a chess engine is good. This goal was accomplished, in an admittedly limited capacity, by utilizing Lichess&#39; open source puzzle database to train a variety of supervised learning technologies including a basic neural network, a decision tree, and a more advanced implementation of a neural network. It was found that, while the decision tree implementation was promisingly consistent over puzzle move length, it&#39;s performance was not convertible to situations which had more features, more data, or more themes due to the high technological cost required when generating tree features. Otherwise, neural networks proved to be superior, if not temperamental, in scenarios involving larger groupings of themes and data.

Given these takeaways, our team recommends to those which wish to further improve upon the strides taken in this report to consider a neural network approach more analytically, and, perhaps, to even adapt a deep learning approach, as our team believes that there&#39;s still yet unexplored potentials in a neural network implementation.
