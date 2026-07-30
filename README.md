# EE15 – Concrete Bridge Deck Crack Detection (Cracked vs Non-Cracked)
GET324 Laboratory Exercise 10 (Mini-Project): Cloud Computing and AI Model
Deployment for Engineering Applications.
## Overview
This project trains a Convolutional Neural Network (CNN) using transfer
learning (MobileNetV3Small) to classify images of concrete bridge deck
surfaces as **cracked** or **non-cracked**, and deploys the trained model
as a Streamlit web application.
## Live App
🔗 https://ee15-crack-detection.streamlit.app
## Dataset
**SDNET2018** (Structural Defects Network 2018), Utah State University.
- Source: https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images
- We used only the **bridge deck subset** (`D/CD` = cracked, `D/UD` = non-cracked)
- Citation: Dorafshan, S., Thomas, R.J., Maguire, M. (2018). "SDNET2018:
  An annotated image dataset for non-contact concrete crack detection
  using deep convolutional neural networks." Data in Brief, 21, 1664–1668.
## Model
- Architecture: MobileNetV3Small (transfer learning) + custom classification head
- Framework: TensorFlow / Keras
- Test accuracy: **92.74%**
- A custom CNN built from scratch was also trained for comparison (88.85% accuracy)
## Project Structure

```
├── app.py                # Streamlit application
├── train_model.py        # Training script
├── requirements.txt      # Python dependencies
├── runtime.txt           # Specifies Python 3.11 for cloud deployment
├── models/
│   └── mobilenetv3_transfer.keras   # Trained model
└── README.md
```
## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
## How to Use the App
1. Open the app (locally or via the live link above)
2. Upload a photo of a concrete bridge deck surface (.jpg or .png)
3. The app displays the prediction (Cracked / Non-Cracked) with a confidence score
## Group Members — EE15

| Name | Registration Number | GitHub Username |
|------|---------------------|------------------|
| Uduakobong Enoh | 22/EG/EE/2047 | denohishim |
| Nyebuk Johnson | 22/EG/EE/2027 | nyebukjay |
| Etimfon Titus | 22/EG/EE/2067 | etirexxie|
| Sinem-favour Udo | 22/EG/EE/1987 | ceenem |
| Ukpong Daniel | 22/EG/EE/2117 | duwem5678-wq |
| Okon Joseph Itoro | 22/EG/EE/2057 | okonj907-arch |
| Victor Augustine | 22/EG/EE/2097 | V-TECHS |
| Eyo Elisha | 22/EG/EE/2037 | elishaeyo1-stack |
## Deployment
Deployed on Streamlit Community Cloud, connected directly to this GitHub
repository. Any push to the `main` branch triggers an automatic redeploy.
