import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Load the model
model = Net()
model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
model.eval()
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)

def predict_digit(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1, keepdim=True).item()

st.title('Digit Recognition App')

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict_digit(image)
    st.write(f'The digit is predicted to be: {label}')

# Add a drawing interface
st.write("Or draw a digit here:")
canvas_result = st_canvas(
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
    st.image(image, caption='Drawn Image.', use_column_width=True)
    st.write("Classifying...")
    label = predict_digit(image)
    st.write(f'The digit is predicted to be: {label}')