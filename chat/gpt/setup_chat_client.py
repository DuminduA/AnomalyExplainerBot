import openai
from django.conf import settings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI


class GPTChat:
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
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
                You are an AI assistant specialized in answering questions politely and professionally
                """

    def __get_system_prompt(self):
        return self.prompt2

    def get_gpt_response(self, user_query: str):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.__get_system_prompt()),
            HumanMessagePromptTemplate.from_template("User Query: {query}")
        ])

        formatted_prompt = prompt.format_messages(query=user_query)  # Pass user query
        response = self.model(formatted_prompt)

        return response




