from django.db import models
# from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User


class CustomUser(models.Model):
  # Add your custom fields here
  user = models.OneToOneField(User, null=True, on_delete=models.CASCADE)
