from django.conf import settings
from langsmith import Client

def get_langsmith_client():
    client = Client(
        api_key=settings.LANGCHAIN_API_KEY
    )

    # try:
    #     project_info = client.create_project(project_name="log-anomaly-detection-langsmith")
    #     client.
    #     print("LangSmith Connected: ✅", project_info)
    #     return client
    # except Exception as e:
    #     print("LangSmith Connection Error: ❌", e)