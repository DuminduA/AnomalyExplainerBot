from django.urls import path

from chatusers import views

urlpatterns = [
    path('login', views.login_user, name="user_login"),
    path('logout', views.logout_user, name="user_logout"),
]