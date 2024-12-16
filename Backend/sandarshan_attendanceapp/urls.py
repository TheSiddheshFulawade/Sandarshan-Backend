from django.urls import path
from . import views

urlpatterns = [
    path('attendance_scan/', views.qr_scan_page, name='qr_scan'),
    path('process_qr_code/', views.process_qr_code, name='process_qr_code'),
    path('process_detection/', views.process_detection, name='process_detection'),
]