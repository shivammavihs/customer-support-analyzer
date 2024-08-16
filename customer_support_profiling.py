from ibm_watson_machine_learning.foundation_models import Model

import streamlit as st

import concurrent.futures


aspect_prompt_mapping = {
    "csat": {
        "label": "Customer Satisfaction",
        "prompt": '''You are a customer service quality analyst. You will be given a Hindi transcription of a customer service call. Your task is to objectively analyze the transcription and assess the service provided by the agent, without any bias.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript, provide a customer satisfaction rating on a scale of 1 to 5 and a short and concise reason for the given rating, based on how satisfied the customer appeared to be at the end of the call. The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Dissatisfied
2 = Dissatisfied
3 = Neutral
4 = Satisfied
5 = Very Satisfied

Provide only the numerical rating and a brief, concise reason for your rating and suggestion if you feel there is any scope of improvement for the agent, if no improvement needed then provide "None" as suggestion.

Output should be in below JSON format:
{{"rating": <customer satisfaction rating>, "reason": "<reason for the rating>","suggestion":"<scope of improvement>"}}

Provide the reason of your rating and suggestion with in 7 to 10 words only.

Result: ''',
    },
    "product_knowledge": {
        "label": "Product Knowledge",
        "prompt": '''You are a customer service quality analyst. You will be given a Hindi transcription of a customer service call. Your task is to objectively analyze the transcription and assess the agent's product knowledge, without any bias.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript, focusing on the agent's product knowledge. Provide a Product Knowledge score on a scale of 1 to 5, based on:
    - Depth of knowledge about products/services
    - Ability to answer questions accurately

The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Poor
2 = Poor
3 = Adequate
4 = Good
5 = Excellent

Provide only the numerical rating, a brief reason for your rating (assessing depth of product knowledge), and a concise suggestion for improvement. If no improvement is needed, provide "None" as the suggestion.

Output should be in below JSON format:
{{"rating": <product knowledge rating>, "reason": "<reason for the rating>", "suggestion":"<area for improvement>"}}

Provide the reason and suggestion in 7 to 10 words each.

Result: ''',
    },
    "empathy": {
        "label": "Empathy",
        "prompt": '''You are a customer service quality analyst. You will be given a Hindi transcription of a customer service call. Your task is to objectively evaluate the agent's empathy and understanding towards the customer.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript and assess the agent's empathy and understanding on a scale of 1 to 5, based on:
    - Demonstrating empathy and understanding of the customer's issue.
    - Showing patience and respect.

The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Provide only the numerical rating, a brief reason for your rating, and a concise suggestion for improvement. If no improvement is needed, provide "None" as the suggestion.

Output should be in below JSON format:
{{"rating": <empathy and understanding rating>, "reason": "<reason for the rating>", "suggestion":"<area for improvement>"}}

Provide the reason and suggestion in 7 to 10 words only.

Result: ''',
    },
    "listening_skills": {
        "label": "Listening Skills",
        "prompt": '''You are a customer service quality analyst. You will be given a Hindi transcription of a customer service call. Your task is to objectively analyze the transcription and assess the agent's listening skills, without any bias.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript, focusing solely on the agent's listening skills. Provide a listening skills rating on a scale of 1 to 5, based on:
    - Active listening and acknowledgment (using phrases like "I understand", "I see", paraphrasing customer's concerns)
    - Not interrupting the customer (allowing them to complete their thoughts)

The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Provide only the numerical rating, a brief reason for your rating (focusing on listening skills), and a concise suggestion for improvement. If no improvement is needed, provide "None" as the suggestion.

Output should be in below JSON format:
{{"rating": <listening skills rating>, "reason": "<reason for the rating>", "suggestion":"<scope of improvement>"}}

Provide the reason for your rating and suggestion in 7 to 10 words only.

Result: ''',
    },
    "comms": {
        "label": "Communication Clarity",
        "prompt": '''You are a customer service quality analyst.  You will be given a Hindi transcription of a customer service call. Your task is to objectively analyze the transcription and assess the agent's communication clarity, without any bias.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript, provide a Communication Clarity rating on a scale of 1 to 5 based on how clearly and effectively the agent communicated. Focus on:
    - Clear and concise language
    - Avoiding jargon and technical terms

The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Unclear
2 = Unclear
3 = Moderately Clear
4 = Clear
5 = Very Clear

Provide only the numerical rating and a brief, concise reason for your rating. Also provide a suggestion if you feel there is any scope for improving the agent's communication clarity. If no improvement is needed, then provide "None" as the suggestion.

Output should be in below JSON format:
{{"rating": <communication clarity rating>, "reason": "<reason for the rating>","suggestion":"<scope of improvement>"}}

Provide the reason for your rating and suggestion in 7 to 10 words only.

Result: ''',
    },
    "call_handling": {
        "label": "Call Handling Skills",
        "prompt": '''You are a customer service quality analyst. You will be given a Hindi transcription of a customer \
service call. Your task is to objectively analyze the transcription and assess the agent's call handling skills, \
without any bias.

Hindi Transcript:
"""{transcription}"""

Analyze the above transcript, focusing on two key aspects:
    - Managing difficult customers effectively
    - Keeping the conversation focused and on track

The rating must be one of [1, 2, 3, 4, 5], where:
1 = Very Unclear
2 = Unclear
3 = Moderately Clear
4 = Clear
5 = Very Clear

Provide only the numerical rating and a brief, concise reason for your rating. Also provide a suggestion if you feel there is any scope for improving the agent's agent's call handling skills. If no improvement is needed, then provide "None" as the suggestion.

Output should be in below JSON format:
{{"rating": <agent's call handling skills rating>, "reason": "<reason for the rating>","suggestion":"<scope of improvement>"}}

Provide the reason for your rating and suggestion in 7 to 10 words only.

Result: ''',
    },
}


class QueryLLM:
    def __init__(self, model_name, parameters, api_key, cloud_url, project_id) -> None:
        self.api_url = cloud_url
        self.api_key = api_key
        self.project_id = project_id

        self.model_id = model_name

        self.parameters = parameters

        self.model = Model(
            model_id=self.model_id,
            params=self.parameters,
            credentials={"url": self.api_url, "apikey": self.api_key},
            project_id=self.project_id,
        )

    def query_llm(self, prompt, stream=False):
        print("=" * 100, "Quering LLM", "=" * 100)
        if stream:
            return self.model.generate_text_stream(prompt)
        else:
            return self.model.generate_text(prompt)

    def detailed_query_llm(self, prompt):
        print("=" * 100, "Quering LLM", "=" * 100)

        return self.model.generate(prompt)


class TransciptAnalyzer:

    def __init__(self, df, api_key, cloud_url, project_id) -> None:
        self.transcription_df = df
        self.model_id = "meta-llama/llama-3-70b-instruct"
        self.llm_params = {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1,
        }
        self.api_key, self.cloud_url, self.project_id = api_key, cloud_url, project_id
        self.transcript = TransciptAnalyzer.format_transcript(df)
        if "llm" not in st.session_state:
            st.session_state["llm"] = QueryLLM(
                self.model_id,
                self.llm_params,
                self.api_key,
                self.cloud_url,
                self.project_id,
            )
        self.llm = st.session_state["llm"]

    @staticmethod
    def format_transcript(df):

        transcript = ""
        for _, row in df.iterrows():
            transcript = f'{transcript}\n{row["speaker_label"]}: {row["text"]}'

        return transcript.strip()

    def validate_aspect(self, label, prompt):
        return [label, self.llm.query_llm(prompt)]

    def analyze_aspects(
        self,
        wx_api_key,
        cloud_url,
        project_id,
    ):
        aspects = list(aspect_prompt_mapping.keys())
        labels: list[str] = [aspect_prompt_mapping[i]["label"] for i in aspects]
        prompts = [
            aspect_prompt_mapping[i]["prompt"].format(transcription=self.transcript)
            for i in aspects
        ]

        a = TransciptAnalyzer(
            self.transcription_df,
            wx_api_key,
            cloud_url,
            project_id,
        )
        with concurrent.futures.ThreadPoolExecutor(6) as executor:
            list_rows = executor.map(a.validate_aspect, labels, prompts)

        responses = list(list_rows)
        return responses
