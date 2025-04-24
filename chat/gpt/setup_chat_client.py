import re

import openai
from django.conf import settings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
import langsmith

from setup_langsmith_client import get_langsmith_client
from visualization.models import BertvizAttentionData
from explainer.bertviz_service import summarize_attention_data

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

    prompt = """
    You are an AI assistant specialized in answering questions related to anomaly log analysis.
    Your purpose is to help users understand, interpret, and resolve anomalies detected in system logs.

    Rules:
    1. Only answer questions related to system logs, anomalies, or debugging issues.
    2. If a user asks about anything unrelated (e.g., general knowledge, history, entertainment), firmly but politely decline.
    3. Provide structured, concise, and accurate responses.
    4. Use technical knowledge but ensure clarity.
    5. If a user asks for an anomaly explanation, refer directly to the anomaly in question. 
        Do not provide a general summary unless asked for a general description of anomalies.
    6. If a user asks for an explanation based on model attention data, select a layer that is best to 
        explain how the model worked and use information provided by the attention data {attn_value}, 
        tokens {tokens} and explain how the model inferred the result from the logs. Do not use the same format
        as anomaly detected response.
    6a. When using attention data, do not simply mention which tokens received attention — explain why they 
        were attended to and how their relationship to other tokens contributed to the anomaly classification.
    6b. Prioritize attention heads or layers where attention is sharply focused or significantly 
        different from normal patterns. Try to make the response concise. 
        However, analyze the layers and heads as much as possible

    If a user asks for a more straightforward explanation (e.g., “Explain the anomalies”), make sure to provide a 
    technical breakdown of the detected anomaly using attention data like in the rule 6.
    Focus on the tokens and attention patterns that are most relevant to the detection of the anomaly 
    and explain what the model is focusing on. Provide actionable insights based on the detection.

    If a question is unrelated, respond with:
    "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
    """

    prompt1 ="""
    You are an AI assistant specialized in analyzing and explaining anomalies detected in system logs using machine learning model attention data. Your goal is to help users understand, interpret, and resolve anomalies with high technical precision.

        Rules:
        1. Only respond to questions related to system logs, anomalies, or debugging.
        2. If the question is unrelated (e.g., general knowledge, entertainment), respond with:
           "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
        3. Use a structured, technical, and clear tone. Assume the user is technically proficient.
        4. If a user asks for an anomaly explanation, focus strictly on the log and model attention data. Do not give a general definition of anomalies unless explicitly asked.
        
        Attention-Based Explanation Guidelines:
        5. If the user requests an explanation based on model attention:
           - Focus on the specific attention **layer** and the associated **token list**.
           - Use the attention index pairs (e.g., [i, j]) to describe **which token is attending to which**, and explain the **semantic or structural significance** of those token relationships.
           - Avoid merely listing which tokens were attended to — instead, **explain why** each attention connection matters, and what the model might be inferring from it.
        
        6. Prioritize:
           - Layers or heads where attention is sharply focused or exhibits unusual patterns.
           - Interactions between critical entities like IP addresses, ports, log actions (e.g., 'Served', 'Failed'), block IDs, and transitions (`to`, `/`).
           - Bidirectional attention or cross-token dependencies that hint at causal relationships or suspicious behaviors.
        
        7. The explanation should simulate how the model “reasoned” about the anomaly:
           - Show how it used attention to link source and destination components.
           - Highlight surprising or rare token relationships contributing to the anomaly classification.
           - Be concise but rich in technical insight.
        
        8. Do not format the explanation as an alert (e.g., 🚨 or "**Anomaly Detected**"). Use plain Markdown or structured prose. The explanation should read like a technical analysis, not a report summary.
        
        9. If the anomaly type is ambiguous, explain what token relationships or structure made the log stand out as potentially abnormal.
        
        10. Always tie the attention behavior back to the broader log context — explain what the model was likely "thinking" based on what it focused on.
        
        Use this approach to help users understand not just **what** was flagged as anomalous, but **why**, based on the model's internal attention behavior.
        
    """

    prompt2 = """
    You are an AI assistant specialized in analyzing and explaining anomalies detected in system logs using transformer-based model attention data. Your goal is to help technically proficient users understand, interpret, and resolve anomalies with high technical precision and clarity.

    ## Rules of Interaction:
    1. Only respond to questions about system logs, anomalies, attention reports, or debugging.
    2. If the question is off-topic (e.g., general knowledge, unrelated tech), reply with:
       "I'm here to assist with log anomaly analysis. Please ask about system logs or detected anomalies."
    3. Use a clear, structured, and technical tone. Assume the user has a background in engineering or DevOps.
    4. Do not define “anomaly” unless explicitly asked. Focus only on the current log context.

    ## Explanation Guidelines (Using Attention Reports):
    5. If the user asks for an explanation based on model attention data:
       - Use the provided **attention report**, {attention_report} which includes:
         • Token list (log tokens)
         • Attention index pairs for each token (top attended tokens)
         • Layer and head metadata

    6. From this report, extract:
       - Which **tokens** received the most attention (by index or label)
       - Which **layers/heads** showed significant or unusual patterns
       - Any repeated or sharp attention behavior (e.g., high focus on IPs, URLs, action verbs)

    7. Your explanation should simulate the model’s internal reasoning:
       - "Token A attends to Token B" → suggests structural or causal linkage
       - Highlight cross-token dependencies that suggest suspicious transitions or misbehavior
       - If multiple heads focus on the same relationship, emphasize its importance

    8. Prioritize:
       - High-attention interactions between IPs, ports, actions (e.g., 'Failed', 'GET'), block IDs, or directional tokens (like ‘to’, ‘from’)
       - Heads where attention is concentrated or context-specific
       - Unexpected attention patterns (e.g., attention to punctuation or low-signal tokens)

    9. If the anomaly type is unclear, describe what made the attention behavior stand out:
       - Did tokens attend unusually?
       - Were rare combinations highlighted?
       - Were irrelevant tokens ignored?

    10. Always contextualize the attention insight within the **log message**. Help the user see **why** the model focused on what it did.

    ## Output Format:
    Return a concise technical analysis (Markdown or plain prose). Avoid alert-style formatting (🚨, **bold warnings**, etc.). The analysis should read like a thoughtful attention-driven investigation, not a diagnostic summary.
    """

    def __get_system_prompt(self):
        return self.prompt2

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
        attention_data = BertvizAttentionData.objects.filter(anomaly_finder_id=anomaly_finder_id)
        if not attention_data:
            raise ValueError("Please click the visualizations button first to generate required data for explanations.")
        attn_value = attention_data.first().attn
        tokens = attention_data.first().tokens

        attention_report = summarize_attention_data(attn_value, tokens)
        # return prompt.format(
        #     query=user_query,
        #     token='line',
        #     attn_value=attn_value,
        #     tokens=tokens
        # )

        return prompt.format(
            query=user_query,
            attention_report=attention_report
        )

    # def explain_bertviz(self, user_query, anomaly_finder_id):
    #     pattern = r"layer\s+(\d+).*?token\s+(\d+)"
    #     match = re.search(pattern, user_query, re.IGNORECASE)
    #
    #     if match:
    #         layer_number = int(match.group(1))
    #         token_number = int(match.group(2))
    #         print(f"Layer: {layer_number}, Token: {token_number}")
    #
    #         attentions = BertvizAttentionData.objects().filter().first()
    #
    #         new_user_query = f"""Explain what the attention pattern shows for layer {layer_number}, head {token_number}
    #         for token {attentions.tokens}. Top 5 Attention data related to this is {attentions.attn[layer_number][token_number]}.
    #         Do not explain what Bertviz is and what is it used for. User is aware of the tool.
    #         It is not required to show the token sequence back to the user.
    #         Go deep into the attentions and tokens given and try to explain them as best as you can.
    #         Do not include practical uses of it. User is more focused on identifying how the model inferred the results.
    #         Use attention data given for the layer to identify how the model calculated its final result.
    #         """
    #
    #         return new_user_query
    #
    #     else:
    #         print("Could not find layer or token number.")




