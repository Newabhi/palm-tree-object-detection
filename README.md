# Palm Tree Detection


This is a project which used the ariel image data of palm tree and using pretrained FR CNN model to detect the palm tree.



# MLOPs Structure
- **Object Detection**
- **Model Performance Metrics**
- **Image Preprocessing**
- **API Integration**
- **Data Versioning**


### Downloading the Dataset

The data is downloaded from Kaggle using kaggle API.

## Model Used

Fasterrcnn_resnet50_fpn, SOTA model is used for palm tree detection.

## Training

The model was trained using the following configuration:

- **Optimizer**: SGD
- **Learning Rate**: 0.01
- **Batch Size**: 16
- **Epochs**: 2
- **Image Size**: 320

The training process was tracked and logged using MLflow. The public MLflow dashboard for the results can be accessed via this [link](https://dagshub.com/aditya.prashant0/my-first-repo.mlflow)

## Metrics, Model Evaluation and Performance

### Metrics

The following metrics were used to assess performance:

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted palm tree counts and the actual counts.
- **Root Mean Squared Error (RMSE)**: The square root of the average squared differences between the predicted and actual palm tree counts.


To run this project locally, follow these steps:

1. **Create a Python environment:**

```bash
python -m venv <env_name>
source <env_name>/bin/activate  # On Windows, use `<env_name>\Scripts\activate`
```

2. **Install the required packages:**

```bash
pip install -r requirements.txt
```


### Training the Model and Managing the Pipeline with DVC

To train the model and manage the entire pipeline, follow these steps:

1. **Run the Training Pipeline**:
    
    ```bash
    dvc repro
    ```
    
    This command will execute the pipeline defined in the `dvc.yaml` file, which includes stages for data ingestion, preprocessing, model training, and evaluation.
    
2. **Train the Model Manually**:
If you prefer to train the model directly without running the pipeline with dvc, use the following command:
    
    ```bash
    python train.py
    ```
    
    This script will load the dataset, preprocess the images, fine-tune the `fasterrcnn_resnet50_fpn` model, and log the training metrics and model artifacts to MLflow.
    

### API

The FastAPI-based API is containerized using Docker and configured to run on GPU-enable platforms for ease of deployment. It provides the following endpoints:

- **`/predict`**: Accepts an image and returns the predicted number of palm trees.
- **`/health`**: Checks the server health
- **`/docs`**: API documentation

### Running the API Locally
To run the API without Docker, follow these steps:
 1. Download the model weight via this link: [palm-tree-counter]()
 2. Place the downloaded model weight in the `models` directory.
 3. Run the API with the following command:
 ```bash
 uvicorn server:app --port 10000
 ```

### Running the API with Docker

To run the API using Docker, follow these steps:

1. **Build the Docker Image:**
    
    In the root directory of your project, run the following command to build the Docker image:
    
    ```bash
    docker build -t palm-tree-counter-api .
    ```
    
2. **Run the Docker Container:**
    
    After building the image, run the container with the following command:
    
    ```bash
    docker run -d -p 10000:10000 palm-tree-counter-api
    ```
    
    This will start the FastAPI server on `http://localhost:10000`.
    
3. **Access the API:**
    
    You can now access the API endpoints:
    
    - **Prediction Endpoint**: `http://localhost:10000/predict`
    - **Health Endpoint**: `http://localhost:10000/health`
    - **Docs Endpoint**: `http://localhost:10000/docs`
    
    The API will serve the model, allowing you to send images and receive the predicted palm tree counts, bounding boxes and confidence scores.
