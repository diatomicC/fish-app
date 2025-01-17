import streamlit as st
import cv2
import base64
from PIL import Image
import io
from openai import OpenAI
import os
from gtts import gTTS
import tempfile
import time

# Initialize OpenAI client - get API key from environment variable or Streamlit secrets
api_key = os.getenv('OPENAI_API_KEY') or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def encode_image(image):
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def text_to_speech(text):
    temp_file = None
    try:
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        
        # Close the file before playing
        temp_file.close()
        
        # Play the audio
        st.audio(temp_file.name, format='audio/mp3')
        
        # Small delay to ensure file is not in use
        time.sleep(0.5)
        
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
    finally:
        # Clean up in finally block
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Ignore deletion errors

def analyze_image(image):
    base64_image = encode_image(image)
    
    try:
        # First request for markdown response
        markdown_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image for fish. Format your response in markdown as follows:

If a fish is present:
# [Fish Species Name > just guess eventhough it is not correct. YOU MUST GUESS]
## Characteristics
[List key physical characteristics very short]

## Habitat
[Describe natural habitat. very short]

## Interesting Facts
[List 2-3 interesting facts. very short]

If no fish is present:
# Image Description
[Describe what you see in the image very short]""",
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

        # Second request for speech response
        speech_response = client.chat.completions.create(
            model="gpt-4o-mini",  # Changed to gpt-4o-mini
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Give me the name of the fish and brief description. very short",
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
            max_tokens=200,
        )

        return (
            markdown_response.choices[0].message.content,
            speech_response.choices[0].message.content
        )
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return "Error occurred while analyzing the image. Please try again.", "An error occurred while analyzing the image."

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

        # Automatically analyze when image is captured/uploaded
        with st.spinner("Analyzing image..."):
            # Get both markdown and speech versions of the analysis
            markdown_result, speech_result = analyze_image(image)
            
            # Display the markdown result
            st.markdown(markdown_result)
            
            # Automatically play the speech version
            text_to_speech(speech_result)

if __name__ == "__main__":
    main() 
