from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.http import HttpResponse, HttpResponseForbidden
from .models import *
from .forms import CreateUserForm


# register new user
def user_register(request):
    if request.user.is_authenticated:
      return redirect("https://creditizens.local/streamlit")

    form = CreateUserForm()

    if request.method == 'POST':
      form = CreateUserForm(request.POST)

      if form.is_valid():
        user = form.save()
        username = form.cleaned_data['username']
        user = authenticate(request, username=username, password=form.cleaned_data['password1'])

        if user is not None:

          login(request, user)
          user_message = form.cleaned_data.get('username')
          print('Account was created for ' + user_message)
          messages.success(request, 'Account was created for ' + user_message)

          return redirect("https://creditizens.local/streamlit")
        else:
          print("Failed to log in. Please try again.")
          messages.error(request, 'Failed to log in. Please try again.')
      else:
        errors = form.errors
        for field, error in errors.items():
          print(f"Error in {field}: {', '.join(error)}")
          messages.error(request, f"Error in {field}: {', '.join(error)}")

    context = {"form": form}
    return render(request, "registeruser.html", context)

# login User
def user_login(request):
    if request.user.is_authenticated:
       return redirect("https://creditizens.local/streamlit")

    if request.method == 'POST':
        username = request.POST.get('username') # we get this from the form (loginuser.html) input name=""
        password = request.POST.get('password') # we get this from the form (loginuser.html) input password=""

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("https://creditizens.local/streamlit")
        else:
            messages.info(request, 'Username OR password is incorrect') # flash a message

    context = {'success': False, 'message': 'Invalid request'}
    return render(request, 'loginuser.html', context)


# logout user
def user_logout(request):
  logout(request)
  return redirect('userauth:user-login')

# redirects to pdf-app
@login_required(login_url='userauth:user-login')
def streamlit_app(request):
  print("redirect to: http://creditizens.local:8002")
  return redirect("http://creditizens.local:8002")


# validate session
def validate_session(request):
  if request.user.is_authenticated:
    print("user authenticated and redirected to pdf app")
    return HttpResponse('User can be redirected to pdf app. is_authenticated=True')
  return HttpResponseForbidden('Access denied. You are not logged in.')
