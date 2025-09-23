#!/usr/bin/env python3
"""
Simple test script for the streaming API endpoint.
Run this after starting the API server to test streaming functionality.
"""

import requests
import json
import sys
import time

def test_streaming():
    """Test the streaming chat endpoint."""
    url = "http://localhost:8080/chat/stream"
    
    payload = {
        "query": "Who is Jayden?",
        "chat_history": ""
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Testing streaming endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\nStreaming response:")
    print("-" * 50)
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        accumulated_response = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "content" in data:
                            content = data["content"]
                            print(content, end="", flush=True)
                            accumulated_response += content
                        elif "done" in data and data["done"]:
                            print("\n" + "-" * 50)
                            print("Stream completed!")
                            break
                        elif "error" in data:
                            print(f"\nError: {data['error']}")
                            return False
                            
                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse JSON: {e}")
                        print(f"Raw line: {line}")
        
        print(f"\nFull response length: {len(accumulated_response)} characters")
        return True
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8080")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False

def test_regular_endpoint():
    """Test the regular (non-streaming) chat endpoint for comparison."""
    url = "http://localhost:8080/chat"
    
    payload = {
        "query": "Who is Jayden?",
        "chat_history": ""
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\nTesting regular endpoint for comparison...")
    print(f"URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload)
        end_time = time.time()
        
        response.raise_for_status()
        data = response.json()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print("Response:")
        print("-" * 50)
        print(data.get("response", "No response"))
        print("-" * 50)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False

if __name__ == "__main__":
    print("Spite AI Streaming Test")
    print("=" * 50)
    
    # Test streaming endpoint
    streaming_success = test_streaming()
    
    # Test regular endpoint for comparison
    regular_success = test_regular_endpoint()
    
    print("\nTest Summary:")
    print(f"Streaming endpoint: {'✓ Success' if streaming_success else '✗ Failed'}")
    print(f"Regular endpoint: {'✓ Success' if regular_success else '✗ Failed'}")
    
    if not (streaming_success or regular_success):
        print("\nTo start the server, run:")
        print("uv run uvicorn src.spite_ai.api_simple:app --host 0.0.0.0 --port 8080")
        sys.exit(1)
