import gradio as gr
import speech_recognition as sr
import threading
import time
import os
import summarizer

# Global variables
is_transcribing = False
current_speaker = "Doctor"
recognizer = sr.Recognizer()
audio_source = None


def transcribe_audio():
    global is_transcribing, current_speaker, audio_source
    with audio_source as source:
        while is_transcribing:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                transcription = f"{timestamp} - {current_speaker}: {text}\n"

                with open("conversation_transcript.txt", "a") as file:
                    file.write(transcription)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print(f"Could not understand {current_speaker}")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


def start_transcription():
    global is_transcribing, audio_source
    if not is_transcribing:
        is_transcribing = True
        audio_source = sr.Microphone()
        threading.Thread(target=transcribe_audio, daemon=True).start()
        return "Transcription started. Speak now."
    return "Transcription is already in progress."


def stop_transcription():
    global is_transcribing
    if is_transcribing:
        is_transcribing = False
        time.sleep(1)  # Give a moment for the transcription thread to finish
        return "Transcription stopped."
    return "No transcription in progress."


def switch_speaker():
    global current_speaker
    current_speaker = "Patient" if current_speaker == "Doctor" else "Doctor"
    return f"Switched to {current_speaker}"


def get_transcript():
    try:
        with open("conversation_transcript.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "No transcript available yet."


def generate_summary():
    try:
        summary = summarizer.main()
        if summary:
            return summary
        else:
            return "Unable to generate summary. Please check the transcript."
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def update_transcript():
    return get_transcript()


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Doctor-Patient Conversation Transcriber and Summarizer")

    with gr.Row():
        start_btn = gr.Button("Start Transcription")
        stop_btn = gr.Button("Stop Transcription")
        switch_btn = gr.Button("Switch Speaker")

    status_output = gr.Textbox(label="Status")

    transcript_output = gr.Textbox(label="Transcript", lines=10)

    summary_btn = gr.Button("Generate Summary")
    summary_output = gr.Textbox(label="Summary", lines=5)

    start_btn.click(start_transcription, outputs=status_output)
    stop_btn.click(stop_transcription, outputs=status_output)
    switch_btn.click(switch_speaker, outputs=status_output)
    summary_btn.click(generate_summary, outputs=summary_output)

    # Add automatic update for transcript
    demo.load(update_transcript, outputs=transcript_output, every=1)

if __name__ == "__main__":
    if os.path.exists("conversation_transcript.txt"):
        os.remove("conversation_transcript.txt")
    demo.launch()