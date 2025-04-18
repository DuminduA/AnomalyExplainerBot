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
    prompt1 = """
                You are an AI assistant specialized in answering questions strictly related to anomaly log analysis.
                Your purpose is to help users understand, interpret, and resolve anomalies detected in system logs.
                
                Rules:
                1. Only answer questions related to system logs, anomalies, or debugging issues.
                2. If a user asks about anything unrelated (e.g., general knowledge, history, entertainment), firmly but politely decline.
                3. Provide structured, concise, and accurate responses.
                4. Use technical knowledge but ensure clarity.
                
                If a question is unrelated, respond with:
                "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
                """

    prompt2 = """
                You are an AI assistant specialized in answering questions politely and professionally. 
                However do not always ask the user for more questions. Just answering the question is enough
                """

    def __get_system_prompt(self):
        return self.prompt2

    @traceable
    def get_gpt_response(self, user_query: str, conversation_history: list):

        messages = []
        for msg in conversation_history:
            if msg["role"] == "bot":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

        if "Explain Bertviz" in user_query:
            user_query = self.explain_bertviz(user_query)

        messages.append(HumanMessage(content=user_query))

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.__get_system_prompt()),
            *messages
        ])

        formatted_prompt = prompt.format_messages(query=user_query)
        response = self.model.invoke(formatted_prompt)

        return response

    def explain_bertviz(self, user_query):
        pattern = r"layer\s+(\d+).*?token\s+(\d+)"
        match = re.search(pattern, user_query, re.IGNORECASE)

        if match:
            layer_number = int(match.group(1))
            token_number = int(match.group(2))
            print(f"Layer: {layer_number}, Token: {token_number}")

            attentions = AttentionData.objects().order_by('-counter').first()

            new_user_query = f"""Explain what the attention pattern shows for layer {layer_number}, head {token_number} 
            for token {attentions.tokens}. Attention data related to this is {attentions.attn[layer_number][token_number]}. 
            Do not explain what Bertviz is and what is it used for. User is aware of the tool. 
            It is not required to show the token sequence back to the user. 
            Go deep into the attentions and tokens given and try to explain them as best as you can. 
            Do not include practical uses of it. User is more focused on identifying how the model inferred the results.
            Use attention data given for the layer to identify how the model calculated its final result.
            """

            return new_user_query

        else:
            print("Could not find layer or token number.")




