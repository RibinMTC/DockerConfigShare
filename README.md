###Predictors Config
    1. Mlsp Aesthetic Score Predictor 
    2. Deepface Emotion Predictor 
    3. Hecate Image Metrics Predictor 
    4. Parnaca Image Metrics Predictor 
    
###Usage
    1. Install Docker(Windows/Mac: Install Docker Desktop)
    2. Drag all images to be predicted, to the prediction_images folder
    3. Select in the input_config.json file under predictors, which features should be extracted(see Predictors Config)
    4. Call docker-compose up --build (from this folder containing the docker-compose.yml file)
    5. Now prediction should be starting. Once prediction is finished, the results are written to the prediction_results.json file
    
