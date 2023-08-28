<h1 align="center">MLOps🚀 - From developement to deployment🧪💥</h1>

#### This repository has been featured in DVC Aug'23 newletter🎉🎉. [Click here](https://www.linkedin.com/pulse/dvc-august-23-community-updates-iterative-ai)

> ***In short for Machine Learning Operations, is a set of practices and methodologies that aim to streamline the deployment, management, and maintenance of machine learning models in production environments. It brings together the principles of DevOps (Development Operations) and applies them specifically to machine learning workflows. The MLOps lifecycle encompasses various stages and processes, ensuring the smooth integration of machine learning models into real-world applications.***

<div align="center">
  <img src="https://github.com/karan842/mlops-best-practices/blob/master/img/mlops-best-practices.png" alt="Banner"/>
</div>

### NEED FOR MLOPS🔮?

Implementing MLOps practices is crucial for several reasons:

1. Reproducibility: MLOps ensures that the entire machine learning pipeline, from data preprocessing to model deployment, is reproducible. This means that the same results can be obtained consistently, facilitating debugging, testing, and collaboration among team members.

2. Scalability: MLOps allows for the seamless scaling of machine learning models across different environments and datasets. It enables efficient deployment and management of models, regardless of the volume of data or the complexity of the infrastructure.

3. Agility: MLOps promotes agility by enabling rapid experimentation, iteration, and deployment of models. It facilitates quick feedback loops, allowing data scientists and engineers to adapt and improve models based on real-world performance and user feedback.

4. Monitoring and Maintenance: 
MLOps ensures continuous monitoring of deployed models, tracking their performance and detecting anomalies. It enables proactive maintenance, including retraining models, updating dependencies, and addressing potential issues promptly.

5. Collaboration: MLOps fosters collaboration among data scientists, engineers, and other stakeholders involved in the machine learning workflow. It establishes standardized practices, tools, and documentation, enabling efficient communication and knowledge sharing.


<div align="center">
  <img src="https://assets-global.website-files.com/5e9aa66fd3886aa2b4ec01ca/630341e16bdfc87f7cd23ee0_ezgif.com-gif-maker%20(1).gif" alt="MLOps GIF" />
</div>

   
This project aims to implement the MLOps (Machine Learning Operations) lifecycle from scratch. The stages involved in the lifecycle include:


### MLOps STAGE 🪜
<details>
<summary>Set Project🐣</summary>

Set up your project environment and version control system for MLOps.

1. Create a Python virtual environment to manage dependencies.
2. Initialize Git and set up your GitHub repository for version control.
3. Install DVC (Data Version Control) for efficient data versioning and storage.
4. Install project dependencies using `requirements.txt`.
5. Write utility scripts for logs, exception handling, and common utilities.

</details>

<details>
<summary>Exploratory Data Analysis📊</summary>

Perform EDA on your data to gain insights and understand statistical properties.

1. Explore the data to understand its distribution and characteristics.
2. Plot charts and graphs to visualize data patterns and relationships.
3. Identify and handle outliers and missing data points.

</details>

<details>
<summary>Data Pipeline🚧</summary>

Create a data ingestion pipeline for data preparation and versioning.

1. Write a data ingestion pipeline to split data into train and test sets.
2. Store the processed data as artifacts for reproducibility.
3. Implement data versioning using DVC for maintaining data integrity.
4. Use the Faker library to generate synthetic data with noise for testing purposes.

</details>

<details>
<summary>Data Transformation🦾</summary>

Perform data transformation tasks to ensure data quality and consistency.

1. Write a script for data transformation, including imputation and outlier detection.
2. Handle class imbalances in the dataset.
3. Implement One-Hot-Encoding and scaling for features.

</details>

<details>
<summary>Model Training🏋️</summary>

Train and tune multiple classification models and track experiments.

1. Train and tune various classification models on the data.
2. Use MLflow for experimentation and tracking model metrics.
3. Log results in the form of JSON to track model performance.

</details>

<details>
<summary>Validation Pipeline✅</summary>

Create a Pydantic pipeline for data preprocessing and validation.

1. Define a Pydantic data model to enforce data validation and types.
2. Implement a pipeline for data preprocessing and validation.
3. Verify the range of values and data types for data integrity.

</details>

<details>
<summary>Create a FastAPI⚡</summary>

Build a FastAPI to make predictions using your trained models.

1. Develop a FastAPI application to serve predictions.
2. Integrate the trained models with the FastAPI endpoint.
3. Provide API documentation using Swagger UI.

</details>

<details>
<summary>Test the API⚗️</summary>

Conduct thorough testing of your FastAPI application.

1. Use Pytest to test different components of the API.
2. Test data types and handle missing input scenarios.
3. Ensure the API responds correctly to various inputs.

</details>

<details>
<summary>Containerization and Orchestration🚢</summary>

Prepare your application for deployment using containers and orchestration.

1. Build a Docker image for your FastAPI application.
2. Push the Docker image to Azure Container Registry (ACR).
3. Test the application locally using Minikube.
4. Deploy the Docker image from ACR to Azure Kubernetes Service (AKS) for production.

</details>

<details>
<summary>CI/CD🔁</summary>

Set up a Continuous Integration and Continuous Deployment pipeline for your application.

1. Configure CI/CD pipeline for automated build and testing.
2. Deploy the application on Azure using CI/CD pipelines.

</details>

### Run by yourself🏃‍♂️
1. Clone the repository:
```bash
git init
git clone https://github.com/karan842/mlops-best-practices.git
```

3. Create a virtual environment
```bash
python -m venv env
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the Flask App
```bash
python app.py
```

6. Run Data and Model pipeline (Enable MLFLOW)
```bash
mlflow ui
dvc init
dvc repro
```

7. Test the application
```bash
pytest
```

### Contribute to it🌱
To make contribution in this project:
- Clone the repository.
- Fork the repository.
- Make changes.
- Create a Pull request.
- Also, publish an issue!
  

### Machine Learning Tool Stack📚
<div align="center">
  <img src="https://fullstackdeeplearning.com/spring2021/lecture-6-notes-media/Infra-Tooling3.png" alt="Infrastructure Tooling" />
</div>

### Acknowledgement📃:
1. Machine Learning in Production (DeepLearning.AI) - Coursera
2. MLOps communities from Discord, Twitter, and LinkedIn
3. Kubernetes, MLFlow, Pytest official documents
4. Microsoft Learning
5. ChatGPT and Bard

### Connect Me🤝:
[Gmail](karanshingde@gmail.com) | [LinkedLin](https://www.linkedin.com/in/karanshingde) | [Twitter](https://www.twitter.com/kuchbhikaran)

