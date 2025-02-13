from django.shortcuts import render
from rest_framework import viewsets


def upload(request):
    return render(request, 'uploader/uploader.html')


class UploaderViewSet(viewsets.ViewSet):
    ...
