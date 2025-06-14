# Object-detection-and--Audio Feedback
This project uses computer vision and audio to help users understand their surroundings. It detects objects using a camera or image and speaks out what it sees — making it especially useful for visually impaired users.
🛠️ Key Features
✅ Detects everyday objects using a pre-trained AI model (like YOLO or MobileNet).

🔊 Gives real-time voice feedback using text-to-speech (TTS).

🎥 Works with a webcam or static images.

💻 Technologies Used
Python

OpenCV – for capturing and processing images

YOLOv5 / MobileNet SSD – for object detection

pyttsx3 / gTTS – for generating audio

🚀 How It Works
The camera or image input is captured using OpenCV.

The AI model detects and labels objects in the frame.

The names of the objects are converted into speech.

The system speaks the object names out loud.

🌟 Use Cases
🧑‍🦯 Assistive tool for the visually impaired

🛡️ Smart surveillance and security systems

🗣️ Voice-enabled object detection for interactive applications


How to run streamlit application:

streamlit run objdetection.py        // streamlit run file_name.py

