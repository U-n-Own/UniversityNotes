
# Autonomous Exercise Voting Algorithm for Pediatric Rehabilitation

**Objective:** The objective of this proposal is to implement an algorithm that autonomously assigns votes to exercises based on the  AHA data initial assignment. The algorithm will utilize Dynamic Time Warping (DTW) or Long Short-Term Memory (LSTM) techniques to track exercises during execution, the main need is to gather relative labels for each exercise, and train a model to recognize exercises. The ultimate goal is to enable patients, particularly pediatric cases with unilateral cerebral palsy, to perform exercises at home under parental supervision while receiving autonomous feedback.
## **Methods:**

1. **Data Collection:**
    
    - Utilize AHA data consisting of sessions with 20 minutes of repeated exercises.
    - Capture relevant features that characterize exercise movements and patterns.
    - This maybe will be the difficult part since we need to gather manually some labelled data.
    - 
2. **Dynamic Time Warping (DTW) or Long Short-Term Memory (LSTM):**
    
    - Apply DTW or LSTM to track exercises during execution.
    - DTW can be employed for measuring the similarity between different instances of exercises.
    - 
    - LSTM, being a recurrent neural network, can capture sequential dependencies in the exercise data so we can try to do the exercise in a different order then gather the indexes and see if this changes influences the score assigned or the indexes.

3. **Autonomous Feedback System:**
    
    - Develop an interface for patients to perform exercises at home, supervised by parents.
    - Integrate the trained model into the system to autonomously provide feedback on the correctness of exercise execution.
    - Feedback can include exercise vote assignments, helping patients and parents track progress.

### Expected Outcomes:

- Creation of an *autonomous exercise voting algorithm* capable of recognizing and providing feedback on exercises performed at home.
- Improved engagement and adherence to rehabilitation routines among pediatric patients with unilateral cerebral palsy.
- Enhanced convenience for parents in supervising their children's exercises.

**Significance:**

- Empowering pediatric patients to engage in rehabilitation exercises at home under parental supervision.
- Reducing the need for frequent clinic visits while maintaining the quality of rehabilitation through autonomous feedback.

## Implementation Plan:

1. **Data Preprocessing:** Prepare the AHA dataset, extract relevant features, and preprocess the data for DTW or LSTM.
2. **Model Development:** Implement DTW or LSTM for exercise tracking, label data, and train a machine learning model for exercise recognition.
3. **Interface Design:** Develop a user-friendly interface for patients and parents to interact with the system during home exercises.
4. **Integration:** Integrate the trained model into the interface to provide autonomous exercise feedback.
5. **Testing and Validation:** Conduct rigorous testing to ensure the accuracy and reliability of the algorithm across diverse scenarios.
6. **Deployment:** Roll out the autonomous exercise voting system for initial trials and gather user feedback for further refinements.

## Conclusion: 

This proposal outlines the development of an innovative autonomous exercise voting algorithm, leveraging DTW or LSTM, to facilitate pediatric rehabilitation at home. The project aims to enhance the quality of care for patients with unilateral cerebral palsy and improve their rehabilitation experience. The successful implementation of this algorithm has the potential to revolutionize remote rehabilitation practices, making them more accessible and effective.