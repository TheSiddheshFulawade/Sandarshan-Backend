from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import *

class RebitAdminUserAdmin(UserAdmin):
    model = RebitAdmin
    list_display = ['username', 'email', 'phone', 'is_staff', 'is_superuser']
    fieldsets = UserAdmin.fieldsets + (
        (None, {'fields': ('phone',)}),
    )

admin.site.register(RebitAdmin, RebitAdminUserAdmin)
admin.site.register(EmployeeUser)
admin.site.register(GuestUser)