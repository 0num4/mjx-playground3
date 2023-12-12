

import fireworks.client
fireworks.client.api_key = "4aYh4cPuZRRiE9OLFBkDrm4MHaGgHrvzl5A1Aeb8CWYFV3MM"
completion = fireworks.client.ChatCompletion.create(
  model="accounts/fireworks/models/elyza-japanese-llama-2-7b-fast-instruct",
  messages=[
    {
      "role": "user",
      "content": "にゃああ",
    }
  ],
  stream=True,
  n=1,
  max_tokens=150,
  temperature=0.1,
  top_p=0.9, 
)