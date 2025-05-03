from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate

def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'forbidden.html', {})
    return render(request, 'login.html', {})

def logout_user(request):
    logout(request)
    return redirect('user_login')



