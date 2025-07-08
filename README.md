An android app that uses machine learning models to predict the likelihood of 
- Diabetes in women
- Heart disease
- Breast cancer
Built with Android studio using Kotlin and tensorflow lite models.

The machine learning models used in this app (for Diabetes, Heart Disease, and Breast Cancer detection) were trained offline by me using datasets like Pima Indians, UCI Heart, and Breast Cancer Wisconsin, and then converted to .tflite format to be used inside the Android application.
These are pre-trained models from the app’s perspective, enabling offline prediction without retraining.

## 🔍 Features

- 🔄 **Three-in-one predictor**: Select between Diabetes, Heart Disease, and Breast Cancer.
- 🤖 Uses **pre-trained ML models** with TensorFlow Lite for fast, on-device predictions.
- 🧪 **Sample Data** autofill to test predictions quickly.
- 🧭 Smooth UI navigation between input forms and disease types.
- 📊 Multi-step input for breast cancer for cleaner UX.
