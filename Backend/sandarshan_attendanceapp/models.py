from django.db import models
from sandarshan_userapp.models import EmployeeUser, GuestUser

# You can add any attendance-specific models here
class Attendance(models.Model):
    user = models.ForeignKey(EmployeeUser, on_delete=models.CASCADE)
    check_in_time = models.DateTimeField(auto_now_add=True)
    check_out_time = models.DateTimeField(null=True, blank=True)
    # Add other fields as needed