import os
import openai
from itertools import groupby

openai.api_key = ""

def resp(text):
  response = openai.Completion.create(
	  engine="text-davinci-003",
	  prompt="You will be a response AI. Respond to the input accordingly with only relevent responses. End each sentence with some variations and synonyms of 'sir'. Your name is PRATIK if anyone asks. Here is the input: " + text,
	  temperature=1,
	  max_tokens=1000,
	  top_p=1,
	  frequency_penalty=1,
	  presence_penalty=1,
	)
  place_holder = int(response.choices[0].text.rfind('\n'))
  result = response.choices[0].text[place_holder : response.choices[0].text.index(response.choices[0].text[-1]) + 1]
  return result