import streamlit as st

from utilities import (
    m4a_to_wav,
    call_speech_to_text,
    process_transcript,
    display_sentiment,
    json_parser,
    display_stars,
)
from customer_support_profiling import TransciptAnalyzer
from sentiment_analysis import analyse_sentiment

from io import BytesIO

st.set_page_config(
    page_title="Customer Support Profiling",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:

    ## getting watson STT credentials
    st.markdown("#### WatsonX STT url")
    st.text_input(
        "url",
        key="url",
        label_visibility="collapsed",
    )

    st.markdown("#### WatsonX STT api key")
    st.text_input(
        "api_key",
        key="api_key",
        label_visibility="collapsed",
        type="password",
    )

    st.markdown("#### WatsonX.ai api key")
    st.text_input(
        "wx_api_key",
        key="wx_api_key",
        label_visibility="collapsed",
    )

    st.markdown("#### WatsonX cloud url")
    st.text_input(
        "cloud_url",
        key="cloud_url",
        label_visibility="collapsed",
    )

    st.markdown("#### WatsonX.ai project id")
    st.text_input(
        "project_id",
        key="project_id",
        label_visibility="collapsed",
    )

st.header("Customer Support Profiling")

## getting the call recording audio file (supports only "m4a" and "wav" format)
st.markdown("#### Upload call recording")
st.file_uploader("file", key="file", type=["m4a", "wav"], label_visibility="collapsed")


def main():
    if st.session_state.file:

        ## playing the uploaded audio file
        st.audio(st.session_state.file)

        audio_file = st.session_state.file

        ## checking the audio format of the recording
        if audio_file.name.split(".")[-1] == "m4a":
            audio_file = m4a_to_wav(audio_file)

            wav_io = BytesIO()
            audio_file.export(wav_io, format="wav")
            wav_io.seek(0)
            audio_file = wav_io

        if (
            st.session_state.url
            and st.session_state.api_key
            and st.session_state.wx_api_key
            and st.session_state.cloud_url
            and st.session_state.project_id
        ):

            with st.spinner("Transcribing the call recording..."):

                st.markdown("#### Transcription")
                place_holder = st.empty()

                response = call_speech_to_text(
                    audio_file, st.session_state.url, st.session_state.api_key
                )
                place_holder.success("Transcription Completed")

                transcription = process_transcript(response)

                place_holder.dataframe(
                    transcription, use_container_width=True, hide_index=True
                )

            st.markdown("### Sentiments")
            col1, col2, col3 = st.columns(3)
            with st.spinner("Analyzing Sentiments..."):
                obj = TransciptAnalyzer(
                    transcription,
                    st.session_state.wx_api_key,
                    st.session_state.cloud_url,
                    st.session_state.project_id,
                )
                sentiments = analyse_sentiment(obj.llm, transcription)
            print(sentiments)
            with col1:
                st.markdown("#### Overall")
                overall_sentiment = display_sentiment(
                    sentiments["overall"]["sentiment"]
                )
                st.markdown(overall_sentiment, unsafe_allow_html=True)
                st.markdown(f'**Reason:** {sentiments["overall"]["reason"]}')
                st.markdown(
                    f'**Scope of Improvement:** {sentiments["overall"]["suggestion"]}'
                )

            with col2:
                st.markdown("#### Agent")
                agent_sentiment = display_sentiment(sentiments["agent"]["sentiment"])
                st.markdown(agent_sentiment, unsafe_allow_html=True)
                st.markdown(f"""**Reason:** {sentiments["agent"]['reason']}""")
                st.markdown(
                    f"""**Scope of Improvement:** {sentiments["agent"]['suggestion']}"""
                )

            with col3:
                st.markdown("#### Customer")
                customer_sentiment = display_sentiment(
                    sentiments["customer"]["sentiment"]
                )
                st.markdown(customer_sentiment, unsafe_allow_html=True)
                st.markdown(f"""**Reason:** {sentiments["customer"]['reason']}""")
                st.markdown(
                    f"""**Scope of Improvement:** {sentiments["customer"]['suggestion']}"""
                )

            with st.spinner("Analyzing for different aspects..."):
                responses = obj.analyze_aspects(
                    st.session_state.wx_api_key,
                    st.session_state.cloud_url,
                    st.session_state.project_id,
                )

            responses = {k: json_parser(v, obj.llm) for k, v in responses}
            st.empty()
            st.markdown("### Customer Agent Support Profile")

            num_rows = (
                len(responses) // 3
                if len(responses) % 3 == 0
                else (len(responses) // 3) + 1
            )
            rows = [st.columns(3) for i in range(num_rows)]
            i = 0
            for row in rows:
                for col in row:
                    col.markdown(f"#### {list(responses.keys())[i]}")
                    star_html = display_stars(
                        responses[list(responses.keys())[i]]["rating"]
                    )
                    col.markdown(star_html, unsafe_allow_html=True)
                    col.markdown(
                        f"""**Reason:** {responses[list(responses.keys())[i]]["reason"]}""",
                        unsafe_allow_html=True,
                    )
                    col.markdown(
                        f"""**Scope of Improvement:** {responses[list(responses.keys())[i]]["suggestion"]}""",
                        unsafe_allow_html=True,
                    )

                    i += 1
        else:
            st.warning("Provide watsonX credentials.")
    else:
        st.warning("Upload the call recording.")


if __name__ == "__main__":
    main()
