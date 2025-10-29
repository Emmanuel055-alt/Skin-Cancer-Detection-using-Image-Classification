# 🧬 Skin Cancer Detection

This project aims to **detect skin cancer from images** using a **deep learning model**.  
It involves training the model, making predictions, evaluating its performance, and running an interactive **Streamlit web app** for real-time classification.

---


Project Structure
models/: Directory containing the trained model.
src/: Directory containing the training script.
app.py: Streamlit application for skin cancer detection.
main.py: Script to train the model.
predict.py: Script to make predictions and evaluate the model.
##  ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

git clone <repository-url>
cd Mini-project_code

### 2️⃣ Create a virtual environment

python -m venv venv
### 3️⃣ Activate the Virtual Environment

**On Windows:**
venv\Scripts\activate
**On macOS/Linux:**
source venv/bin/activate
### 4️⃣ Install Dependencies
pip install -r requirements.txt
 ## 🧠 Training the Model
To train the model, run the main.py script:

python main.py

## 🔍 Making Predictions
To make predictions using the trained model, run the predict.py script:

python predict.py
## 💻 Running the Streamlit App
To run the Streamlit app, use the following command:

streamlit run app.py
This will start the Streamlit server, and you can view the app in your web browser at the URL provided in the terminal output (usually http://localhost:8501).

## 🧩 Dependencies
streamlit
tensorflow
numpy
scikit-learn
These dependencies are listed in the requirements.txt file and can be installed using pip install -r requirements.txt.
## 📊 Model Overview
Architecture: Convolutional Neural Network (CNN)

Input: Skin lesion images

Output: Binary classification — Benign or Malignant

Evaluation Metrics: Accuracy, Precision, Recall, F1-score
