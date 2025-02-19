from django.urls import path

from chatusers import views

urlpatterns = [
    path('', views.login_user, name="login"),
    path('', views.login_user, name="login"),
]