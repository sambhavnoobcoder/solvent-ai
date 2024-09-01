import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def read_conversation(file_path):
    """Read the conversation from the transcript file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def clean_conversation(conversation):
    """Clean the conversation to remove timestamps and extraneous details."""
    return "\n".join(line.split(" - ", 1)[-1] for line in conversation.splitlines())

def generate_summary(conversation, model_name="facebook/bart-large-cnn"):
    """Generate a detailed and natural summary using an open-source model like BART."""
    try:
        # Load the BART model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Clean the conversation
        clean_conversation_text = clean_conversation(conversation)

        # Prepare the input for summarization
        inputs = tokenizer(f"summarize: {clean_conversation_text}", return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=150,
            min_length=50,
            num_beams=4,  # Increase the number of beams for beam search
            length_penalty=2.0,
            no_repeat_ngram_size=3,  # Prevent repeating n-grams
            do_sample=False  # Disable sampling for deterministic output
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Generated summary: {summary}")  # Debugging output

        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        sys.exit(1)

def refine_summary_with_flan_t5(summary, model_name="google/flan-t5-large"):
    """Refine the summary to make it sound more natural using a language model like Flan-T5."""
    try:
        # Load the Flan-T5 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Prepare the input for the model
        prompt = f"Paraphrase this in a more natural, conversational style: {summary}"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate the refined summary
        response_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=100,
            num_beams=4,  # Use beam search for more coherent output
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            do_sample=False
        )

        refined_summary = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Debugging: print the raw response
        print(f"Refined summary from Flan-T5: {refined_summary}")

        return refined_summary.strip()
    except Exception as e:
        print(f"Error refining summary: {e}")
        sys.exit(1)

def save_summary(summary, output_file="conversation_summary.txt"):
    """Save the generated summary to a file."""
    try:
        with open(output_file, "w") as file:
            file.write(summary)
        print(f"\nSummary saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving summary: {e}")

def main():
    transcript_file = "conversation_transcript.txt"

    if not os.path.exists(transcript_file):
        print(f"Error: '{transcript_file}' not found. Please check the file path.")
        sys.exit(1)

    # Read and clean the conversation
    conversation = read_conversation(transcript_file)

    print("Generating summary...")
    summary = generate_summary(conversation)

    print("Refining summary with Flan-T5...")
    refined_summary = refine_summary_with_flan_t5(summary)

    print("\nRefined summary of the conversation:")
    print(refined_summary)

    # Save the refined summary to a file
    save_summary(refined_summary)

if __name__ == "__main__":
    main()
