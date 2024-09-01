import speech_recognition as sr
import threading
import time
import os

# Global variables
current_speaker = "Doctor"
is_listening = True
recognizer = sr.Recognizer()


def recognize_speech_realtime(source):
    global current_speaker, is_listening
    while is_listening:
        try:
            audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            transcription = f"{timestamp} - {current_speaker}: {text}\n"
            print(transcription, end='')

            # Save transcription to file
            with open("conversation_transcript.txt", "a") as file:
                file.write(transcription)
        except sr.WaitTimeoutError:
            pass  # No speech detected, continue listening
        except sr.UnknownValueError:
            print(f"Could not understand {current_speaker}")
        except sr.RequestError as e:
            print(f"Could not request results; {0}".format(e))


def switch_speaker():
    global current_speaker
    current_speaker = "Patient" if current_speaker == "Doctor" else "Doctor"
    print(f"\nSwitched to {current_speaker}")


def handle_user_input():
    global is_listening
    while is_listening:
        user_input = input()
        if user_input.lower() == 's':
            switch_speaker()
        elif user_input.lower() == 'q':
            is_listening = False
            print("\nEnding conversation...")
        else:
            print("Invalid input. Press 's' to switch speakers or 'q' to quit.")


def transcribe_conversation():
    global is_listening

    # Clear previous transcript
    if os.path.exists("conversation_transcript.txt"):
        os.remove("conversation_transcript.txt")

    print("Starting conversation transcription.")
    print("Press 's' to switch speakers or 'q' to quit at any time.")

    with sr.Microphone() as source:
        # Adjust for ambient noise
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        # Start the speech recognition in a separate thread
        speech_thread = threading.Thread(target=recognize_speech_realtime, args=(source,))
        speech_thread.start()

        # Handle user input in the main thread
        handle_user_input()

        # Wait for the speech recognition thread to finish
        speech_thread.join()

    print("\nConversation ended. Transcript saved to 'conversation_transcript.txt'")


# Run the conversation transcription
transcribe_conversation()