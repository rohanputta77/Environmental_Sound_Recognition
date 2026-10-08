# Sound Classification Request Flow

This diagram illustrates exactly what happens when a user uploads an audio file for prediction. It tracks the request as it travels through each file in the Layered Architecture.

```mermaid
sequenceDiagram
    autonumber
    
    actor User as User (Frontend)
    participant Main as backend/main.py<br/>(Manager)
    participant Endpoint as api/endpoints/predict.py<br/>(Waiter)
    participant Schema as schemas/predict.py<br/>(Order Form)
    participant Service as services/prediction_service.py<br/>(Chef)
    participant Repo as repositories/model_repository.py<br/>(Barista)
    participant Model as models/model.pth<br/>(Machine)

    %% Request hits the server
    User->>Main: Uploads audio file (POST /predict)
    Main->>Endpoint: Routes traffic to predict endpoint
    
    %% Endpoint handles initial validation and saving
    Endpoint->>Endpoint: Checks if file exists
    Endpoint->>Endpoint: Saves temp audio to uploads/ folder
    
    %% Hand off to Service layer
    Endpoint->>Service: Calls process_audio_prediction(file_path)
    
    %% Service delegates ML to Repository
    Service->>Repo: Calls model_repo.predict(file_path)
    
    %% Repository does the heavy lifting
    Repo->>Model: Loads trained AI weights
    Model-->>Repo: Weights loaded into memory
    Repo->>Repo: Librosa: Convert audio to Mel-Spectrogram
    Repo->>Repo: PyTorch: Run ResNet18 + SVM inference
    Repo-->>Service: Returns predicted class (e.g., "Dog")
    
    %% Service cleans up
    Service->>Service: finally: Deletes temp audio file
    Service-->>Endpoint: Returns predicted class
    
    %% Endpoint formats and returns response
    Endpoint->>Schema: Validates shape (PredictionResponse)
    Schema-->>Endpoint: Approved (Enforces {"prediction": "Dog"})
    Endpoint-->>User: Sends JSON Response (200 OK)
```
