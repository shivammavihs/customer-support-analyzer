from customer_support_profiling import TransciptAnalyzer
import concurrent.futures
from utilities import json_parser, extract_json


def execute_prompt(prompt, label, llm):
    print(f"{prompt=}")
    response = llm.query_llm(prompt)
    print(f"{response=}")
    response = extract_json(response)
    print(f"1{response=}")
    response = json_parser(response, llm)
    print(label, response)
    return [label, response]


def analyse_sentiment(llm, df):

    sentiment_prompt = '''You are an AI assistant tasked with analyzing the sentiment of a customer support call transcript in Hindi and providing a concise evaluation in JSON format. The sentiment classification should be one of the following: Positive, Negative, or Neutral.

While classifying the sentiment, consider multiple aspects that are essential for good customer support, from both the agent's and customer's perspectives, for example:

  - Politeness, professionalism, clarity of communication, problem-solving ability, product knowledge and expertise of the agent.
  - Politeness, clarity of communication, patience, able to provide complete information, understanding nature of the customer. 

Provide the output in below JSON format:
{{
  "sentiment": "<overall sentiment of the transcript>",
  "reason": "<reason for the sentiment>",
  "suggestion": "<scope of improvement for the agent, or 'None' if no improvement is needed>"
}}

Here is the Hindi transcript of the customer support call:

Hindi Transcript: """{transcript}"""

Note: Provide only the JSON output, without any additional text. And the reason for your rating and suggestion should be with in 7 to 10 words only.

JSON Output: '''

    agent_sentiment_prompt = '''You are an AI assistant tasked with analyzing the sentiment of a hindi transcript of the agent's responses only from a customer support call and providing a concise evaluation in JSON format. The sentiment classification should be one of the following: Positive, Negative, or Neutral.

While classifying the sentiment, consider multiple aspects that are essential for good customer support from the agent's perspective only, for example:
  - Politeness, professionalism, clarity of communication, problem-solving ability, product knowledge and expertise.

Provide the output in below JSON format:
{{
  "sentiment": "<sentiment of the agent's responses>",
  "reason": "<reason for the sentiment>",
  "suggestion": "<scope of improvement for the agent, or 'None' if no improvement is needed>"
}}

Here is the Hindi transcript of the agent's responses in the customer support call:

Hindi Transcript: """{transcript}"""

Note: Provide only the JSON output, without any additional text. And the reason for your rating and suggestion should be with in 7 to 10 words only.

JSON Output: '''

    customer_sentiment_prompt = '''You are an AI assistant tasked with analyzing the sentiment of a hindi transcript of the customers's responses only from a customer support call and providing a concise evaluation in JSON format. The sentiment classification should be one of the following: Positive, Negative, or Neutral.

While classifying the sentiment, consider multiple aspects that are essential for good customer support from the customer's perspective only, for example:
  - Politeness, professionalism, clarity of communication, problem-solving ability, product knowledge and expertise.

Provide the output in below JSON format:
{{
  "sentiment": "<sentiment of the customer's responses>",
  "reason": "<reason for the sentiment>",
  "suggestion": "<scope of improvement, or 'None' if no improvement is needed>"
}}

Here is the Hindi transcript of the customer's responses in the customer support call:

Hindi Transcript: """{transcript}"""

Note: Provide only the JSON output, without any additional text. And the reason for your rating and suggestion should be with in 7 to 10 words only.

JSON Output: '''

    prompts = [
        sentiment_prompt.format(transcript=TransciptAnalyzer.format_transcript(df)),
        agent_sentiment_prompt.format(
            transcript=TransciptAnalyzer.format_transcript(
                df[df["speaker_label"] == "agent"]
            )
        ),
        customer_sentiment_prompt.format(
            transcript=TransciptAnalyzer.format_transcript(
                df[df["speaker_label"] == "customer"]
            )
        ),
    ]
    labels = ["Overall", "Agent", "Customer"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        responses = executor.map(
            execute_prompt, prompts, labels, [llm for i in range(len(labels))]
        )

    responses = {k.lower(): v for k, v in list(responses)}
    return responses
