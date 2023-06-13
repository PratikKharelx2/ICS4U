import os
import openai

openai.api_key = "MY KEY :)"

def func(text):
  response = openai.Completion.create(
	  engine="text-davinci-003",
	  prompt="Act as jarvis from iron man but your name will be Pratik. Finish each sentence with 'sir'. Only return your response, without your name. Here is the input: " + text,
	  temperature=1,
	  max_tokens=1000,
	  top_p=1,
	  frequency_penalty=1,
	  presence_penalty=1,
	)
  resp = response.choices[0].text.replace("\n", "")
  return(resp)
