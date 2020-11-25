###Predictors Config
1. Mlsp Aesthetic Score Predictor. \
   Output: An aesthetic score between 0 and 10.
2. Deepface Emotion Predictor. \
   Output: Predicts an emotion like surprised, happy, sad etc. and none if no face or no emotion is detected.
3. Hecate Image Metrics Predictor.\
   Output: For a full documentation of the output see [here](https://github.com/yahoo/hecate/blob/master/include/hecate/image_metrics.hpp).
4. Parnaca Image Metrics Predictor. \
   Output: Each predicted metric is between -1 and 1, except the _TotalScore_, which is between 0 and 1.
    
###Usage
1. Install Docker (Windows/Mac: Install Docker Desktop)
2. Drag all images to be predicted, to the _prediction_images_ folder
3. Select in the _input_config.json_ file under _predictors_, which features should be extracted(see Predictors Config)
4. Call _docker-compose up --build_ (from the folder containing the docker-compose.yml file)
5. Now prediction should be starting. Once prediction is finished, the results are written to the _prediction_results.json_ file
    
