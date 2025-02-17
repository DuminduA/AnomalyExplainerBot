from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from chat.gpt.setup_chat_client import GPTChat

def home(request):
    return render(request, 'home/home.html')


class ChatBotViewSet(viewsets.ViewSet):
    client = GPTChat()
    @action(detail=False, methods=['post'], name="chat-with-gpt", url_path="chat-with-gpt")
    def get_response(self, request):
        user_message = request.data

        response = self.client.get_gpt_response(user_message)

        return Response({"bot_message": response.content})