### Predictors Config 
1. Mlsp Aesthetic Score Predictor. \
   Output: An aesthetic score between 0 and 10.
2. Deepface Emotion Predictor. \
   Output: Predicts an emotion like surprised, happy, sad etc. and none if no face or no emotion is detected.
3. Hecate Image Metrics Predictor.\
   Output: For a full documentation of the output see [here](https://github.com/yahoo/hecate/blob/master/include/hecate/image_metrics.hpp).
4. Parnaca Image Metrics Predictor. \
   Output: Each predicted metric is between -1 and 1, except the _TotalScore_, which is between 0 and 1. More info in their [project page](https://www.ics.uci.edu/~skong2/aesthetics.html).
5. PAZ Emotion Predictor. \
   Output:  Predicts an emotion with the corresponding bounding box coordinates(xMin, yMin, xMax, yMax) or returns an empty list if nothing is detected. More info in their [project page](https://github.com/oarriaga/paz).
6. FER Emotion Predictor. \
   Output:  Predicts emotions(angry, disgust, fear, happy, neutral, sad, surprise) with a corresponding confidence score(0-1) and the bounding box coordinates(xMin, yMin, xMax, yMax) or returns an empty list if nothing is detected. More info in their [project page](https://github.com/justinshenk/fer).
     
### Usage
1. Install Docker (Windows/Mac: Install Docker Desktop)
2. Drag all images to be predicted, to the _prediction_images_ folder
3. Select in the ``input_config.json`` file under _predictors_, which features should be extracted(see Predictors Config)
4. From the folder containing the ``docker-compose.yml`` file, run:
```
docker-compose up --build
```
5. Now prediction should be starting. Once prediction is finished, the results are written to the ``prediction_results.json`` file

For more details on how to add a custom aesthetic predictor to the api, see [here](https://docs.google.com/document/d/1hDjqn07FveOCTLcdRhFALy0dsiGLQw783rGL7uP4Ckw/edit?usp=sharing).

### Evaluation
We use the file ``preprocessing_datasets.py`` to prepare the data. Especially we use it to divide the data in test, train and val folders, and to generate pandas datafames with the data.

We then use the file ``process_results.py`` to get correlations between the extracted features and the aesthetic scores.

You can see some of the results in this [google doc](https://docs.google.com/document/d/1L2JBULEQu3bWDCpukInqkYeG1ndYQQ6cb0YXURxVq0Y/edit?usp=sharing).


