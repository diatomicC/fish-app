import streamlit as st
import cv2
import base64
from PIL import Image
import io
from openai import OpenAI
import os

# Initialize OpenAI client - get API key from environment variable or Streamlit secrets
api_key = os.getenv('OPENAI_API_KEY') or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def encode_image(image):
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image(image):
    base64_image = encode_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is there a fish in this image? If yes, please identify the specific type of fish and provide a detailed description including its characteristics, habitat, and interesting facts. If no fish is present, simply respond with 'No fish detected in this image.'",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        st.error("Please make sure you have access to GPT-4 Vision API in your OpenAI account")
        return "Error occurred while analyzing the image. Please try again."

def main():
    st.title("Fish Species Identifier")
    st.write("Take a picture or upload an image to identify fish species!")

    # Add file uploader and camera input
    img_file = st.camera_input("Take a picture")
    
    if img_file is None:
        img_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])

    if img_file is not None:
        # Display the captured image
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a button to trigger analysis
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Get the analysis result
                result = analyze_image(image)
                
                # Display the result
                st.write("### Analysis Result:")
                st.write(result)

if __name__ == "__main__":
    main() 