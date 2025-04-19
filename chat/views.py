import http.client

from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from chat.gpt.setup_chat_client import GPTChat

def home(request):
    if request.user.is_authenticated:
        return render(request, 'home/home.html')
    else:
        return render(request, 'forbidden.html')


class ChatBotViewSet(viewsets.ViewSet):
    client = GPTChat()
    @action(detail=False, methods=['post'], name="chat-with-gpt", url_path="chat-with-gpt")
    def get_response(self, request):
        if request.user.is_authenticated:
            try:
                user_message = request.data
                conversation_history = request.session.get("chat_history", [])

                anomaly_finder_id = request.session.get("anomaly_finder_id")

                response = self.client.get_gpt_response(user_message, conversation_history, anomaly_finder_id)

                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "bot", "content": response.content})
                request.session['chat_history'] = conversation_history

                return Response({"bot_message": response.content})
            except ValueError as e:
                return Response({"bot_message": str(e)})
        else:
            return Response({"message": "Forbidden"}, status=http.client.FORBIDDEN)

    @action(detail=False, methods=['delete'], name="clear-chat", url_path="clear-chat")
    def clear_history(self, request):
        if request.user.is_authenticated:
            request.session["chat_history"] = []
            return Response({"success": True})
        else:
            return Response({"message": "Forbidden"}, status=http.client.FORBIDDEN)