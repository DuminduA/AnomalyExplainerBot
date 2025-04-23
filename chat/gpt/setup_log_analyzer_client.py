import openai
from django.conf import settings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable

from langsmith.wrappers import wrap_openai

@traceable
class GPTAnomalyAnalyzer:

    client = wrap_openai(openai.OpenAI(api_key=settings.OPENAI_API_KEY))
    model = ChatOpenAI(model='gpt-4o', temperature=0)

    def __get_system_prompt(self):
        return """
                You are an intelligent log analysis assistant that helps users understand and resolve anomaly logs efficiently.
                Your responses should be structured, concise, and action-oriented.
                
                Instructions:
                1. Identify the anomaly and state its type.
                2. Provide context, including log source and severity.
                3. Summarize the issue in simple, human-readable language.
                4. Suggest a possible cause based on log patterns.
                5. Recommend an action to resolve the issue.
                6. Include a relevant log snippet.
                7. Be concise and professional.
                8. Prioritize clarity and actionability. If the anomaly type is unclear, suggest general troubleshooting steps.
                
                Format:
                🚨 Anomaly Detected: 
                Log Source:
                Severity:
                Issue Summary:
                Possible Cause:
                Recommended Action:
                
                🔍 Log Snippet:
                `{log_data}`
                """

    def get_gpt_response(self, log_data: list):
        results = []

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.__get_system_prompt()),
            HumanMessagePromptTemplate.from_template("Log Data: {log_data}")
        ])

        for log in log_data:
            formatted_prompt = prompt.format_messages(log_data=log)
            response = self.model.invoke(formatted_prompt)
            results.append(response.content)

        return results

