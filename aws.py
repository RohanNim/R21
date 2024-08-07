import os
import requests
import openai
import speech_recognition as sr
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import subprocess
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters
from elevenlabs.core import ApiError
# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

recognizer = sr.Recognizer()

def generate_response(transcript):
    prompt = """You are an AI version of President Joe Biden. You answer questions with clarity, empathy, and a focus on positive outcomes. Your responses are detailed and well-structured, reflecting the style and tone of President Biden. Take personal accountability where necessary and emphasize your past achievements and future plans.

Examples:

Question: "What are your plans for improving healthcare in America?"

Real Biden Response: "Well, let me be clear. The healthcare system needs an overhaul. We must protect the Affordable Care Act and expand access to Medicare. Everyone should have the right to affordable healthcare, and my administration will work tirelessly to ensure that."

AI Biden Response: "Thank you for that important question. Our healthcare system faces numerous challenges that need immediate attention. My administration is committed to strengthening and expanding the Affordable Care Act to ensure that every American has access to affordable healthcare. We will work on reducing prescription drug costs, increasing funding for mental health services, and expanding Medicare. Our goal is to create a healthcare system that is fair, efficient, and accessible to all."

Question: "How will you address the economic impact of the pandemic?"

Real Biden Response: "We've got to build back better. This pandemic has devastated our economy, and we need a comprehensive plan to recover. This includes direct financial assistance to families, support for small businesses, and investing in infrastructure."

AI Biden Response: "The economic impact of the pandemic has been profound, affecting families, businesses, and communities nationwide. My administration has a clear plan to build back better, which includes providing direct financial relief to families, supporting small businesses with grants and loans, and making significant investments in infrastructure projects to create jobs and stimulate economic growth. We will also focus on education and job training to prepare our workforce for the future. Together, we can emerge from this crisis stronger and more resilient."

Question: "What do you say to Americans concerned about your age and mental fitness?"

Real Biden Response: "Well, look, after that debate, I did ten major events in a row, including until 02:00 in the morning. After that debate, I did events in North Carolina. I did events in Georgia. Did events like this today. Large crowds, overwhelming response, no slipping. And so I just had a bad night. I don't know why I, and how."

AI Biden Response: "I understand that my age and mental fitness are valid concerns for many Americans. Let me assure you, my commitment to serving this country remains unwavering. Since that debate, I have participated in numerous events, from North Carolina to Georgia, meeting with large crowds and receiving overwhelming support. I continue to work tirelessly every day, just as I have for the past three and a half years. My focus is on leading this country towards a brighter future, addressing the challenges we face, and delivering on the promises we've made. Together, we will achieve great things.
above are only example for referencne but Give me responses/reply in only 120 characters strictly. """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )

    return response['choices'][0]['message']['content']

def text_to_video(text, user_id):
    try:
        audio_generator = client.text_to_speech.convert(
            voice_id="zSovZJHF0b6l3tv5AZE9",
            optimize_streaming_latency="1",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        audio_file_path = f"{user_id}_response.mp3"
        with open(audio_file_path, "wb") as audio_file:
            for chunk in audio_generator:
                audio_file.write(chunk)
    
        # Run Wave2Lip inference
        subprocess.run(["python", "inference.py", "--checkpoint_path", "checkpoints/wav2lip_gan.pth", "--face", "biden.mp4", "--audio", audio_file_path,"--resize_factor","2"])

        result_video_path = "results/result_voice.mp4"
        return result_video_path

    except ApiError as api_err:
        print(f"API Error: {api_err.body['detail']['message']}")
        return None

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hi! Send me a text or audio message and I will reply with a video.')

async def handle_text(update: Update, context: CallbackContext) -> None:
    user_text = update.message.text
    await update.message.reply_text('Processing Started')
    ai_response = generate_response(user_text)
    video_path = text_to_video(ai_response, update.message.chat_id)
    if video_path:
        await update.message.reply_video(video_path)
    else:
        await update.message.reply_text('Sorry, there was an error processing your request.')

async def handle_voice(update: Update, context: CallbackContext) -> None:
    try:
        await update.message.reply_text('Processing Started')
        # Get the file object
        file = await update.message.voice.get_file()

        # Download the file
        file_url = file.file_path
        response = requests.get(file_url)
        with open('voice.ogg', 'wb') as f:
            f.write(response.content)

        # Convert the file to WAV format
        subprocess.run(["ffmpeg","-y" , "-i", "voice.ogg", "voice.wav"])

        # Recognize the audio and generate a response
        with sr.AudioFile('voice.wav') as source:
            audio = recognizer.record(source)
        transcript = recognizer.recognize_google(audio)
        ai_response = generate_response(transcript)
        video_path = text_to_video(ai_response, update.message.chat_id)

        if video_path:
            await update.message.reply_video(video_path)
        else:
            await update.message.reply_text('Sorry, there was an error processing your request.')
    except sr.UnknownValueError:
        await update.message.reply_text('Sorry, I could not understand the audio.')
    except sr.RequestError as e:
        await update.message.reply_text(f"Error with the request; {e}")
    except Exception as e:
        await update.message.reply_text(f"An unexpected error occurred: {e}")
def main() -> None:
    # Initialize the bot application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Start polling
    application.run_polling()

if __name__ == '__main__':
    main()






