import streamlit as st
import os
from dotenv import load_dotenv
import speech_recognition as sr
import openai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import elevenlabs
import subprocess  # Import subprocess module

# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

recognizer = sr.Recognizer()

def generate_response(transcript):
    # Define the prompt
    prompt = """You are an AI version of President Joe Biden. You answer questions with clarity, empathy, and a focus on positive outcomes. Your responses are detailed and well-structured, reflecting the style and tone of President Biden. Take personal accountability where necessary and emphasize your past achievements and future plans. Give me responses in 200 characters only.

Examples:

Question: "What are your plans for improving healthcare in America?"

Real Biden Response: "Well, let me be clear. The healthcare system needs an overhaul. We must protect the Affordable Care Act and expand access to Medicare. Everyone should have the right to affordable healthcare, and my administration will work tirelessly to ensure that."

AI Biden Response: "Thank you for that important question. Our healthcare system faces numerous challenges that need immediate attention. My administration is committed to strengthening and expanding the Affordable Care Act to ensure that every American has access to affordable healthcare. We will work on reducing prescription drug costs, increasing funding for mental health services, and expanding Medicare. Our goal is to create a healthcare system that is fair, efficient, and accessible to all."

Question: "How will you address the economic impact of the pandemic?"

Real Biden Response: "We've got to build back better. This pandemic has devastated our economy, and we need a comprehensive plan to recover. This includes direct financial assistance to families, support for small businesses, and investing in infrastructure."

AI Biden Response: "The economic impact of the pandemic has been profound, affecting families, businesses, and communities nationwide. My administration has a clear plan to build back better, which includes providing direct financial relief to families, supporting small businesses with grants and loans, and making significant investments in infrastructure projects to create jobs and stimulate economic growth. We will also focus on education and job training to prepare our workforce for the future. Together, we can emerge from this crisis stronger and more resilient."

Question: "What do you say to Americans concerned about your age and mental fitness?"

Real Biden Response: "Well, look, after that debate, I did ten major events in a row, including until 02:00 in the morning. After that debate, I did events in North Carolina. I did events in Georgia. Did events like this today. Large crowds, overwhelming response, no slipping. And so I just had a bad night. I don't know why I, and how."

AI Biden Response: "I understand that my age and mental fitness are valid concerns for many Americans. Let me assure you, my commitment to serving this country remains unwavering. Since that debate, I have participated in numerous events, from North Carolina to Georgia, meeting with large crowds and receiving overwhelming support. I continue to work tirelessly every day, just as I have for the past three and a half years. My focus is on leading this country towards a brighter future, addressing the challenges we face, and delivering on the promises we've made. Together, we will achieve great things."""
    
    # Generate response using OpenAI GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )

    return response['choices'][0]['message']['content']

def handle_conversation():
    st.title("Techi-The Bot")

    option = st.radio("Select input method:", ("Voice", "Text"))

    if option == "Voice":
        if st.button("Ask a question"):
            try:
                with sr.Microphone() as source:
                    st.write("Listening...")
                    audio = recognizer.listen(source, timeout=3)  # Adjust timeout as needed (e.g., 3 seconds)

                # Convert speech to text
                transcript = recognizer.recognize_google(audio)
                st.write("You said:", transcript)

                # Generate AI response
                text = generate_response(transcript)

                # Convert text to speech using ElevenLabs
                try:
                    audio_generator = client.text_to_speech.convert(
                        voice_id="6tWO8Oz4iafOIoOnQy8Z",  # Adam pre-made voice
                        optimize_streaming_latency="1",
                        output_format="mp3_22050_32",
                        text=text,
                        model_id="eleven_multilingual_v2",  # Use the turbo model for low latency
                        voice_settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=0.75,
                            style=0.0,
                            use_speaker_boost=True,
                        ),
                    )

                    # Save the audio stream to a file
                    audio_file_path = "response.mp3"
                    with open(audio_file_path, "wb") as audio_file:
                        for chunk in audio_generator:
                            audio_file.write(chunk)

                    # Display the audio player on Streamlit UI
                    st.audio(audio_file_path, format='audio/mp3')

                    # Run Wave2Lip inference
                    subprocess.run(["python", "inference.py", "--checkpoint_path", "checkpoints/wav2lip_gan.pth", "--face", "biden1.mp4", "--audio", audio_file_path])

                except elevenlabs.core.api_error.ApiError as api_err:
                    st.write(f"API Error: {api_err.body['detail']['message']}")
                    if 'status' in api_err.body['detail'] and api_err.body['detail']['status'] == 'detected_captcha_voice':
                        st.write("Voice verification required. Please verify your voice.")
                        # Add any required steps to handle captcha/voice verification here

                # Display AI response
                st.write("AI:", text)

            except sr.UnknownValueError:
                st.write("Could not understand audio")
            except sr.RequestError as e:
                st.write(f"Error with the request; {e}")
            except KeyboardInterrupt:
                st.write("Conversation ended.")

    else:
        question = st.text_input("Type your question here:")
        if st.button("Submit"):
            # Generate AI response
            text = generate_response(question)

            # Convert text to speech using ElevenLabs
            try:
                audio_generator = client.text_to_speech.convert(
                    voice_id="6tWO8Oz4iafOIoOnQy8Z",  # Adam pre-made voice
                    optimize_streaming_latency="1",
                    output_format="mp3_22050_32",
                    text=text,
                    model_id="eleven_multilingual_v2",  # Use the turbo model for low latency
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        use_speaker_boost=True,
                    ),
                )

                # Save the audio stream to a file
                audio_file_path = "response.mp3"
                with open(audio_file_path, "wb") as audio_file:
                    for chunk in audio_generator:
                        audio_file.write(chunk)

                # Display the audio player on Streamlit UI
                st.audio(audio_file_path, format='audio/mp3')

                # Run Wave2Lip inference
                subprocess.run(["python", "inference.py", "--checkpoint_path", "checkpoints/wav2lip_gan.pth", "--face", "Biden.mp4", "--audio", audio_file_path])

            except elevenlabs.core.api_error.ApiError as api_err:
                st.write(f"API Error: {api_err.body['detail']['message']}")
                if 'status' in api_err.body['detail'] and api_err.body['detail']['status'] == 'detected_captcha_voice':
                    st.write("Voice verification required. Please verify your voice.")
                    # Add any required steps to handle captcha/voice verification here

            # Display AI response
            st.write("AI:", text)

if __name__ == "__main__":
    handle_conversation()
