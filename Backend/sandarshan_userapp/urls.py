from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='sandarshand_landing'),
    path('admin_register/', views.user_registration, name='rebit_admin_register'),
    path('login/', views.user_login, name='rebit_login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('employee/register/', views.employee_registration, name='employee_registration'),
    path('guest/register/', views.guest_registration, name='guest_registration'),

]
