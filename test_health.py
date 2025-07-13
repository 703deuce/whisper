#!/usr/bin/env python3
"""
Simple health check test for WhisperX API server
"""
import requests
import json
import sys
import time

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8000/healthcheck", timeout=10)
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Server status: {data['status']}")
            print(f"Device: {data['device']}")
            print(f"Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        print(f"Root endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data['message']}")
            print(f"Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"Root endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"Root endpoint error: {e}")
        return False

def test_models_endpoint():
    """Test the models list endpoint"""
    try:
        response = requests.get("http://localhost:8000/models/list", timeout=10)
        print(f"Models endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Available models: {len(data['data'])}")
            return True
        else:
            print(f"Models endpoint failed: {response.text}")
            return False
    except Exception as e:
        print(f"Models endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("Testing WhisperX API server endpoints...")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    tests = [
        ("Root endpoint", test_root_endpoint),
        ("Health check", test_health_endpoint),
        ("Models list", test_models_endpoint)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            print(f"✓ {test_name} passed")
            passed += 1
        else:
            print(f"✗ {test_name} failed")
    
    print(f"\n--- Results ---")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! 🎉")
        sys.exit(0)
    else:
        print("Some tests failed 😞")
        sys.exit(1) 