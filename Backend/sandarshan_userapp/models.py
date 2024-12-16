from django.db import models
from django.contrib.auth.models import AbstractUser

class RebitAdmin(AbstractUser):
    phone = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return self.username
    
class EmployeeUser(models.Model):
    name = models.CharField(max_length=100)
    address = models.TextField()
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20)
    access_expiry = models.DateTimeField()
    office_location = models.CharField(max_length=100)
    room_access = models.TextField()  # You can store room access as a comma-separated list or JSON
    
    def __str__(self):
        return self.name

class GuestUser(models.Model):
    name = models.CharField(max_length=100)
    address = models.TextField()
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    reason_for_visit = models.TextField()
    access_expiry = models.DateTimeField()
    office_location = models.CharField(max_length=100)
    room_access = models.TextField()  # You can store room access as a comma-separated list or JSON
    
    def __str__(self):
        return self.name