from django.shortcuts import render
from django.http import JsonResponse
from pyzbar.pyzbar import decode
from sandarshan_userapp.models import *
from .detection_service import IntegratedDetectionService 
import cv2
import base64
import numpy as np
import json

# Initialize detection service once
detection_service = IntegratedDetectionService()

def qr_scan_page(request):
    """Render the QR code scanning page"""
    return render(request, 'attendance/qr_scan.html')

def process_qr_code(request):
    """
    Advanced QR code processing with multiple recognition strategies
    """
    if request.method == 'POST':
        image_data = request.POST.get('image')
        
        try:
            # Decode base64 image
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Read image with OpenCV
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Decode QR code
            decoded_objects = decode(img)
            
            if decoded_objects:
                # Extract QR code data
                qr_data = decoded_objects[0].data.decode('utf-8')
                
                # Multiple search strategies
                results = search_user_strategies(qr_data)
                
                if results:
                    return JsonResponse({
                        'success': True, 
                        **results
                    })
                
                return JsonResponse({
                    'success': False, 
                    'message': 'No matching user found'
                })
            
            else:
                return JsonResponse({
                    'success': False, 
                    'message': 'No QR code detected'
                })
        
        except Exception as e:
            return JsonResponse({
                'success': False, 
                'message': f'Error processing QR code: {str(e)}'
            })
    
    return JsonResponse({
        'success': False, 
        'message': 'Invalid request method'
    })

def search_user_strategies(qr_data):
    """
    Multiple strategies to find user based on QR code data
    """
    from .models import EmployeeUser, GuestUser
    
    # Strategy 1: Exact Name Match
    employee = EmployeeUser.objects.filter(name=qr_data).first()
    if employee:
        return {
            'name': employee.name, 
            'type': 'Employee',
            'email': employee.email,
            'office_location': employee.office_location
        }
    
    guest = GuestUser.objects.filter(name=qr_data).first()
    if guest:
        return {
            'name': guest.name, 
            'type': 'Guest',
            'email': guest.email,
            'office_location': guest.office_location
        }
    
    # Strategy 2: Partial Name Match
    employee = EmployeeUser.objects.filter(name__icontains=qr_data).first()
    if employee:
        return {
            'name': employee.name, 
            'type': 'Employee',
            'email': employee.email,
            'office_location': employee.office_location
        }
    
    guest = GuestUser.objects.filter(name__icontains=qr_data).first()
    if guest:
        return {
            'name': guest.name, 
            'type': 'Guest',
            'email': guest.email,
            'office_location': guest.office_location
        }
    
    # Strategy 3: JSON or Complex Data Parsing
    try:
        # Try parsing as JSON
        data = json.loads(qr_data)
        
        # Search by email or other identifiers
        if 'email' in data:
            employee = EmployeeUser.objects.filter(email=data['email']).first()
            if employee:
                return {
                    'name': employee.name, 
                    'type': 'Employee',
                    'email': employee.email,
                    'office_location': employee.office_location
                }
            
            guest = GuestUser.objects.filter(email=data['email']).first()
            if guest:
                return {
                    'name': guest.name, 
                    'type': 'Guest',
                    'email': guest.email,
                    'office_location': guest.office_location
                }
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None

def process_detection(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        
        try:
            # Decode base64 image
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Read image with OpenCV
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process detection
            results = detection_service.process_frame(img)
            
            # Prepare response
            response_data = {
                'success': True,
                'faces': [{
                    'name': face.get('name', 'Unknown'),
                    'confidence': float(face.get('confidence', 0)),
                    'bbox': face.get('bbox')  # Include full bbox coordinates
                } for face in results.get('faces', [])],
                'objects': [{
                    'class_name': obj.get('class_name'),
                    'confidence': float(obj.get('confidence', 0)),
                    'bbox': obj.get('bbox')  # Include full bbox coordinates
                } for obj in results.get('objects', [])]
            }
            
            return JsonResponse(response_data)
        
        except Exception as e:
            return JsonResponse({
                'success': False, 
                'message': f'Error processing detection: {str(e)}'
            })
    
    return JsonResponse({
        'success': False, 
        'message': 'Invalid request method'
    })