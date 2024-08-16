import streamlit as st

from pydub import AudioSegment

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1

import pandas as pd

import os
import requests
import re
import json


def m4a_to_wav(input_file):
    audio = AudioSegment.from_file(input_file, format="m4a")
    return audio


@st.cache_data(show_spinner=False)
def call_speech_to_text(audio_file, url, api_key):

    api_endpoint = f"{url}/v1/recognize"

    params = {
        "model": "hi-IN_Telephony",
        # "timestamps": "true",
        "speaker_labels": "true",
        "background_audio_suppression": "0.5",
        "end_of_phrase_silence_time": "1.0",
        "speech_detector_sensitivity": "0.55",
        "smart_formatting": "true",
        "smart_formatting_version": "2",
    }
    # The request headers
    headers = {
        "Content-Type": "audio/wav",
    }

    response = requests.post(
        api_endpoint,
        headers=headers,
        auth=("apikey", api_key),
        data=audio_file,
        params=params,
    )

    return dict(response.json())


@st.cache_data(show_spinner=False)
def process_transcript(stt_response):

    transcript_timestamps = []

    for i in stt_response["results"]:

        transcript_timestamps.extend(i["alternatives"][0]["timestamps"])


    speaker = ""
    timestamps = []
    start = "y"
    final_time = ""
    for i in stt_response["speaker_labels"]:
        if speaker != i["speaker"]:
            if start == "y":
                start = "n"
            else:
                timestamps.append([f"speaker {speaker}", start_time, final_time])
            speaker = i["speaker"]
            start_time = i["from"]
        final_time = i["to"]
    else:
        timestamps.append([f"speaker {speaker}", start_time, final_time])


    sentence = ""
    text_no = 0
    transcription = []
    for i in timestamps:
        speaker = i[0]
        start = i[1]
        end = i[2]
        for j in transcript_timestamps[text_no:]:
            text_no += 1
            if j[1] >= start and j[2] <= end:
                sentence = sentence + " " + j[0]
            else:
                break

        transcription.append([speaker, start, end, sentence])
        sentence = j[0]

    df = pd.DataFrame(transcription, columns=["speaker_label", "start", "end", "text"])

    agent_label = list(df["speaker_label"])[0]

    df["speaker_label"] = df["speaker_label"].apply(
        lambda v: "agent" if v == agent_label else "customer"
    )

    return df


def display_sentiment(sentiment):
    if sentiment.lower() == "positive":
        color = "green"
    elif sentiment.lower() == "negative":
        color = "red"
    else:  # assuming the only other option is 'neutral'
        color = "orange"

    return f'<span style="color:{color}; font-size: 24px;">{sentiment}</span>'


def extract_json(text):
    pattern = r"\{([\s\S]*)\}"
    matches = re.findall(pattern, text)
    return "{" + matches[0] + "}"


def json_parser(llm_response, llm):
    json_text = extract_json(llm_response)
    try:
        json_obj = json.loads(json_text)
    except Exception as e:
        print(e)
        print("correcting JSON")
        prompt = '''You are a JSON formatter. You will be given an invalid JSON string and the Python error encountered when trying to load it using json.loads(). Your job is to correct the invalid JSON string by considering the error message and return the correct JSON only as output, no additional text.  

Invalid JSON String: """{json_text}"""

Python error message: """{error_msg}"""

Corrected JSON String: '''
        corrected_json = llm.query_llm(prompt.format(json_text=json_text, error_msg=e))
        json_obj = json.loads(extract_json(corrected_json))

    return json_obj


def display_stars(
    rating, max_stars=5, size=24, filled_color="#FFD700", empty_color="#DDDDDD"
):
    filled_star = f'<span style="font-size:{size}px; color:{filled_color};">★</span>'
    empty_star = f'<span style="font-size:{size}px; color:{empty_color};">☆</span>'
    stars = filled_star * rating + empty_star * (max_stars - rating)
    return stars
