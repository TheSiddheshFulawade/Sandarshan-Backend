from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.conf import settings
from django.utils import timezone
from .models import *
import os
import qrcode




def user_registration(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        phone = request.POST.get('phone')

        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'registration.html')

        if RebitAdmin.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return render(request, 'registration.html')

        try:
            user = RebitAdmin.objects.create_user(username=username, email=email, password=password, phone=phone)
            messages.success(request, 'Registration successful.')
            return redirect('rebit_login')
        except Exception as e:
            messages.error(request, f'Registration failed: {str(e)}')
            return render(request, 'registration.html')

    return render(request, 'registration.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None and isinstance(user, RebitAdmin):
            login(request, user)
            messages.success(request, 'Login successful.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid credentials.')

    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    messages.success(request, 'Logged out successfully.')
    return redirect('rebit_login')

def dashboard(request):
    if not request.user.is_authenticated or not isinstance(request.user, RebitAdmin):
        return redirect('rebit_login')
    return render(request, 'admin/dashboard.html')

def generate_qr_code(name):
    """Generate QR code for a given name"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(name)
    qr.make(fit=True)
    
    # Generate a unique filename
    filename = f'qr_{name.replace(" ", "_")}.png'
    filepath = os.path.join(settings.MEDIA_ROOT, 'qr_codes', filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filepath)
    
    return f'/media/qr_codes/{filename}'

def employee_registration(request):
    if request.method == 'POST':
        try:
            # Collect form data
            name = request.POST.get('name')
            address = request.POST.get('address')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            access_expiry = request.POST.get('access_expiry')
            office_location = request.POST.get('office_location')
            room_access = request.POST.get('room_access')
            
            # Convert access_expiry to datetime
            access_expiry = timezone.datetime.strptime(access_expiry, '%Y-%m-%d')
            
            # Create employee user
            employee = EmployeeUser.objects.create(
                name=name,
                address=address,
                email=email,
                phone=phone,
                access_expiry=access_expiry,
                office_location=office_location,
                room_access=room_access
            )
            
            # Generate QR code
            qr_code_path = generate_qr_code(name)
            
            return render(request, 'admin/employee_qr.html', {'qr_code': qr_code_path, 'employee': employee})
        
        except Exception as e:
            messages.error(request, f'Registration failed: {str(e)}')
            return render(request, 'admin/employee_registration.html')
    
    return render(request, 'admin/employee_registration.html')

def guest_registration(request):
    if request.method == 'POST':
        try:
            # Collect form data
            name = request.POST.get('name')
            address = request.POST.get('address')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            reason_for_visit = request.POST.get('reason_for_visit')
            access_expiry = request.POST.get('access_expiry')
            office_location = request.POST.get('office_location')
            room_access = request.POST.get('room_access')
            
            # Convert access_expiry to datetime
            access_expiry = timezone.datetime.strptime(access_expiry, '%Y-%m-%d')
            
            # Create guest user
            guest = GuestUser.objects.create(
                name=name,
                address=address,
                email=email,
                phone=phone,
                reason_for_visit=reason_for_visit,
                access_expiry=access_expiry,
                office_location=office_location,
                room_access=room_access
            )
            
            # Generate QR code
            qr_code_path = generate_qr_code(name)
            
            return render(request, 'admin/guest_qr.html', {'qr_code': qr_code_path, 'guest': guest})
        
        except Exception as e:
            messages.error(request, f'Registration failed: {str(e)}')
            return render(request, 'admin/guest_registration.html')
    
    return render(request, 'admin/guest_registration.html')

def landing(request):
    return render(request, 'landing.html')

