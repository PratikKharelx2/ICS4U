# Import os library for interacting with the operating system
import os
# Import openai library for interacting with OpenAI's API
import openai

# Set OpenAI API key using environment variable
openai.api_key = "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"

# Define func function to generate text using OpenAI's API
def func(text):
    # Create a completion using OpenAI's API
    response = openai.Completion.create(
        # Set engine to use for completion
        engine="text-davinci-003",
        # Set prompt for completion
        prompt="Act as jarvis from iron man but your name will be Pratik. Finish each sentence with 'sir'. Only return your response, without your name. Here is the input: " + text,
        # Set temperature for completion
        temperature=1,
        # Set maximum number of tokens to generate
        max_tokens=1000,
        # Set top_p for completion
        top_p=1,
        # Set frequency penalty for completion
        frequency_penalty=1,
        # Set presence penalty for completion
        presence_penalty=1,
    )
    # Get text from first choice of response
    resp = response.choices[0].text.replace("\n", "")
    # Return generated text
    return(resp)
