from django.urls import path
from . import views

app_name = "userauth"

urlpatterns = [
    path('user-register', views.user_register, name='user-register'),
    path('user-login', views.user_login, name='user-login'),
    path('streamlit', views.streamlit_app, name='streamlit-app'),
    path('user-logout', views.user_logout, name='user-logout'),
    path('validate_session', views.validate_session, name='validate-session'),
]
