import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io

# Define the model architecture
class DualInputCNN(nn.Module):
    def __init__(self, hand_classes=25, face_classes=6):
        super(DualInputCNN, self).__init__()
        self.hand_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.face_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.Dropout(0.2)
        )
        # Adjusting these layers to match the expected structure in the saved state dictionary
        self.fc_hand = nn.Sequential(nn.Linear(128, hand_classes))
        self.fc_face = nn.Sequential(nn.Linear(128, face_classes))

    def forward(self, hand_x, face_x):
        hand_features = self.hand_cnn(hand_x).view(hand_x.size(0), -1)
        face_features = self.face_cnn(face_x).view(face_x.size(0), -1)
        combined_features = torch.cat([hand_features, face_features], dim=1)
        output = self.fc(combined_features)
        return self.fc_hand(output), self.fc_face(output)

# Initialize and load the model
model = DualInputCNN()
model.load_state_dict(torch.load('best_fusion_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)


# Define the mappings for hand gestures and face expressions
hand_gesture_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

face_expression_map = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# Streamlit interface
st.title("Gesture and Expression Prediction")
uploaded_file_hand = st.file_uploader("Choose a hand image...", type=["png", "jpg", "jpeg"])
uploaded_file_face = st.file_uploader("Choose a face image...", type=["png", "jpg", "jpeg"])

if st.button("Predict"):
    if uploaded_file_hand is not None and uploaded_file_face is not None:
        image_hand = Image.open(io.BytesIO(uploaded_file_hand.read())).convert('L')
        image_face = Image.open(io.BytesIO(uploaded_file_face.read())).convert('L')
        
        tensor_hand = transform_image(image_hand).unsqueeze(0)
        tensor_face = transform_image(image_face).unsqueeze(0)
        
        with torch.no_grad():
            hand_output, face_output = model(tensor_hand, tensor_face)
            hand_prediction = hand_output.argmax(1).item()
            face_prediction = face_output.argmax(1).item()
        
        predicted_alphabet = hand_gesture_map.get(hand_prediction, 'Unknown')
        predicted_expression = face_expression_map.get(face_prediction, 'Unknown')
        
        st.write(f'Predicted alphabet: {predicted_alphabet}')
        st.write(f'Predicted expression: {predicted_expression}')
    else:
        st.error("Please upload both hand and face images.")