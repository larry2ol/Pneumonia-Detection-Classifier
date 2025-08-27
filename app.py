import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# 2. Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self, image_shape):
        super(CNNModel, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(image_shape[0], 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Convolutional Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Convolutional Block 3
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten and FC with Global Average Pooling and a single Linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(512, 1) # Assuming binary classification output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn1_2(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.conv4(x))
        x = self.bn2_2(x)
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = self.bn3(x)
        x = F.relu(self.conv6(x))
        x = self.bn3_2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_final(x)
        return x

# 3. Define a function to load the trained model
@st.cache_resource
def load_model(model_path):
    model = CNNModel(image_shape=(3, 224, 224))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 4. Define a function to preprocess an uploaded image
def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image

# 5. Define a function to make a prediction
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = torch.sigmoid(outputs) > 0.5
    return prediction.item()

# 6. Set up the Streamlit application interface
st.title("Pneumonia Detection from Chest X-ray Images")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# 7. Add conditional logic to handle the uploaded file
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_container_width=True)

    # Load the model
    model_path = 'best_model.pth'  # Adjust the path if necessary
    model = load_model(model_path)

    # Preprocess the image
    image_tensor = preprocess_image(uploaded_file)

    # Make a prediction
    prediction = predict(model, image_tensor)

    # Display the result
    if prediction == 1:
        st.write("Prediction: Pneumonia")
    else:
        st.write("Prediction: Normal")
