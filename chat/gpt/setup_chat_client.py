import re

import openai
from django.conf import settings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
import langsmith

from setup_langsmith_client import get_langsmith_client
from visualization.models import AttentionData

langsmith.debug = True

class GPTChat:
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    langsmith_cl = get_langsmith_client()
    model = ChatOpenAI(model='gpt-4o', temperature=0)
    # prompt1 = """
    #             You are an AI assistant specialized in answering questions related to anomaly log analysis.
    #             Your purpose is to help users understand, interpret, and resolve anomalies detected in system logs.
    #
    #             Rules:
    #             1. Only answer questions related to system logs, anomalies, or debugging issues.
    #             2. If a user asks about anything unrelated (e.g., general knowledge, history, entertainment), firmly but politely decline.
    #             3. Provide structured, concise, and accurate responses.
    #             4. Use technical knowledge but ensure clarity.
    #             5. However, always try to provide some answer. Do not go into default message unless absolutely necessary.
    #             6. If a user asks about explainability of the detected anomalies, formulate your answer according to the following,
    #                 Explain what the attention pattern shows for layer {layer_number}, head {token_number}
    #                 for token {tokens}. Attention data related to this is {attn_value}.
    #                 Do not explain what Bertviz is and what is it used for. User is aware of the tool.
    #                 It is not required to show the token sequence back to the user.
    #                 Go deep into the attentions and tokens given and try to explain them as best as you can.
    #                 Do not include practical uses of it. User is more focused on identifying how the model inferred the results.
    #                 Use attention data given for the layer to identify how the model calculated its final result.
    #
    #             If a question is unrelated, respond with:
    #             "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
    #             """

    # prompt2 = """
    #             You are an AI assistant specialized in answering questions politely and professionally.
    #             However do not always ask the user for more questions. Just answering the question is enough
    #             """

    prompt1 = """
    You are an AI assistant specialized in answering questions related to anomaly log analysis.
    Your purpose is to help users understand, interpret, and resolve anomalies detected in system logs.

    Rules:
    1. Only answer questions related to system logs, anomalies, or debugging issues.
    2. If a user asks about anything unrelated (e.g., general knowledge, history, entertainment), firmly but politely decline.
    3. Provide structured, concise, and accurate responses.
    4. Use technical knowledge but ensure clarity.
    5. If a user asks for an anomaly explanation, refer directly to the anomaly in question. Do not provide a general summary unless asked for a general description of anomalies.
    6. If a user asks for an explanation based on model attention data, select a layer that is best to explain how the model worked and use information provided by the attention data {attn_value}, tokens {tokens} and explain how the model inferred the result from the logs.

    If a user asks for a more straightforward explanation (e.g., “Explain the anomalies”), make sure to provide a 
    technical breakdown of the detected anomaly using attention data like in the rule 6.
    Focus on the tokens and attention patterns that are most relevant to the detection of the anomaly 
    and explain what the model is focusing on. Provide actionable insights based on the detection.

    If a question is unrelated, respond with:
    "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
    """

    def __get_system_prompt(self):
        return self.prompt1

    @traceable
    def get_gpt_response(self, user_query: str, conversation_history: list, anomaly_finder_id: str):

        messages = []
        for msg in conversation_history:
            if msg["role"] == "bot":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_query))

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.__get_system_prompt()),
            *messages
        ])

        formatted_prompt = self.get_formatted_prompt(anomaly_finder_id, prompt, user_query)
        response = self.model.invoke(formatted_prompt)

        return response

    def get_formatted_prompt(self, anomaly_finder_id, prompt, user_query):
        attention_data = AttentionData.objects.filter(anomaly_finder_id=anomaly_finder_id).first()
        if not attention_data:
            raise ValueError("No attentions data in the database")
        attn_value = attention_data.attn[3][5]
        tokens = attention_data.tokens

        return prompt.format(
            query=user_query,
            token='line',
            attn_value=attn_value,
            tokens=tokens
        )

    def explain_bertviz(self, user_query, anomaly_finder_id):
        pattern = r"layer\s+(\d+).*?token\s+(\d+)"
        match = re.search(pattern, user_query, re.IGNORECASE)

        if match:
            layer_number = int(match.group(1))
            token_number = int(match.group(2))
            print(f"Layer: {layer_number}, Token: {token_number}")

            attentions = AttentionData.objects().filter().first()

            new_user_query = f"""Explain what the attention pattern shows for layer {layer_number}, head {token_number} 
            for token {attentions.tokens}. Top 5 Attention data related to this is {attentions.attn[layer_number][token_number]}. 
            Do not explain what Bertviz is and what is it used for. User is aware of the tool. 
            It is not required to show the token sequence back to the user. 
            Go deep into the attentions and tokens given and try to explain them as best as you can. 
            Do not include practical uses of it. User is more focused on identifying how the model inferred the results.
            Use attention data given for the layer to identify how the model calculated its final result.
            """

            return new_user_query

        else:
            print("Could not find layer or token number.")




