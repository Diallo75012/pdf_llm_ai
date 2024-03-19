from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe


class CreateUserForm(UserCreationForm):
  username = forms.CharField(
    max_length=100,
    widget=forms.TextInput(attrs={'placeholder': 'Username'}),
  )
  password1 = forms.CharField(
    max_length=100,
    widget=forms.PasswordInput(attrs={'placeholder': 'Password'}),
    help_text=mark_safe("<div class='password-help'>Password must contain:</div>"
                        "<div class='password-help'>- At least 8 characters</div>"
                        "<div class='password-help'>- At least one uppercase letter</div>"
                        "<div class='password-help'>- At least one lowercase letter</div>"
                        "<div class='password-help'>- At least one digit</div>")
  )
  password2 = forms.CharField(
    max_length=100,
    widget=forms.PasswordInput(attrs={'placeholder': 'Repeat password'}),
  )
  class Meta:
    model = User
    fields = ('username', 'password1', 'password2')
