# Predicting-User-Engagement-in-Video-Games-on-Steam

**Project Description:**
In this project, I developed and implemented a recommender system to predict user engagement with video games on Steam, focusing on two main tasks: predicting whether a user would play a game and estimating the time a user would spend playing a game. The project utilized a combination of matrix factorization techniques and Bayesian Personalized Ranking (BPR), along with popularity-based models, to achieve accurate predictions.

**Technology Stack:**
- **Data Handling and Scripting:** Python, pandas
- **Machine Learning Algorithms:** Matrix Factorization, Bayesian Personalized Ranking (BPR)
- **Libraries and Tools:** scikit-learn, scipy, implicit

**Key Accomplishments:**

**1. Playtime Prediction:**
   - **Data Preprocessing:** Extracted and processed data from a gzipped JSON file, splitting it into training and validation sets to ensure robust model evaluation.
   - **Bias-Corrected Matrix Factorization:** Developed a matrix factorization model with user and game biases to predict the logarithmically transformed playtime. The model iteratively updated global average, user biases, and game biases to minimize prediction error.
   - **Parameter Tuning:** Optimized the regularization parameter to balance model complexity and accuracy, improving the prediction of playtime for both seen and unseen user-game pairs.
   - **Output Generation:** Generated predictions for the test set and saved them in the required format for submission, ensuring the model's outputs were consistent and ready for evaluation.

**2. Play Prediction:**
   - **Negative Sampling:** Created a negative sample set to balance the dataset, improving the robustness of the play prediction model by ensuring it could distinguish between played and unplayed games.
   - **Bayesian Personalized Ranking (BPR):** Implemented the BPR model to learn latent factors for users and games, which were then used to predict whether a user would play a specific game. The model was trained on a sparse matrix representation of user-game interactions.
   - **Hybrid Model:** Combined the BPR-based prediction with a popularity-based model to improve prediction accuracy, particularly for popular games.
   - **Prediction Logic:** Developed a logic that considers both BPR scores and game popularity, refining the threshold to ensure accurate binary predictions (play or not play).

**Performance Optimization and Evaluation:**
   - **Cross-Validation:** Split the dataset into training and validation sets to prevent overfitting and to evaluate the model’s generalization performance.
   - **Model Iteration:** Iteratively refined the model by adjusting hyperparameters and incorporating additional data features, resulting in improved accuracy on the validation set.
   - **Leaderboard Performance:** Regularly uploaded predictions to the competition leaderboard to benchmark performance and make necessary adjustments to the model.

**Key Outcomes:**
   - **Accurate Playtime Prediction:** Successfully developed a model that predicts the logarithmic transformation of playtime with minimal error, outperforming baseline models.
   - **Improved Play Prediction:** Enhanced the accuracy of predicting whether a user would play a game by combining collaborative filtering techniques with a popularity-based approach.
   - **Scalability and Adaptability:** The models developed are scalable and can be adapted to similar recommendation tasks, ensuring they are versatile for future projects or datasets.
