An end-to-end Machine Learning project that predicts student performance based on academic and demographic factors. The project follows production-style ML pipelines and serves predictions through a Flask web application.

Workflow
Data Ingestion – Load and split raw data into training and testing sets
Data Transformation – Preprocess data using pipelines (encoding, scaling)
Model Training – Train and select the best-performing ML model
Prediction Pipeline – Generate predictions using saved model & preprocessor
Web App – User-friendly prediction via Flask interface

Key Components
app.py – Flask application (entry point)
data_ingestion.py – Data loading and splitting
data_transformation.py – Preprocessing pipelines
model_trainer.py – Model training and evaluation
train_pipeline.py – End-to-end training workflow
predict_pipeline.py – Inference workflow
artifacts/ – Saved model and preprocessor

Tech Stack
Python, Pandas, NumPy, Scikit-learn, Flask, HTML/CSS