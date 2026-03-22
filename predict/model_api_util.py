import argparse
import base64
import json
import requests
import os
import sys

def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_html_file(html_path):
    """Read an HTML file and return base64-encoded content."""
    if not os.path.exists(html_path):
        print(f"HTML file not found: {html_path}")
        sys.exit(1)
    with open(html_path, "r", encoding="utf-8") as f:
        return base64.b64encode(f.read().encode("utf-8")).decode("utf-8")

def call_api(api_name, base_url, method="GET", params=None):
    """Generic API caller."""
    url = f"{base_url.rstrip('/')}/{api_name}"
    try:
        if method.upper() == "GET":
            res = requests.get(url, json=params)
        elif method.upper() == "POST":
            res = requests.post(url, json=params)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        return res.json() if res.ok else {"error": res.text, "status": res.status_code}
    except Exception as e:
        return {"error": str(e)}

def get_params(api_name, args):
    """Return parameters based on API name."""
    if api_name == "predict":
        return {
            "MESSAGE": args.message,
            "IMAGE": image_to_base64(args.image)
        }
    elif api_name == "predictWithMessage":
        return {
            "MESSAGE": args.message
        }
    elif api_name == "predictWithImage":
        return {
            "IMAGE": image_to_base64(args.image)
        }
    elif api_name == "learnWithMessage":
        return {
            "MESSAGE": args.message,
            "CLASSIFICATION": args.classification
        }
    elif api_name == "learnWithImage":
        return {
            "IMAGE": image_to_base64(args.image),
            "CLASSIFICATION": args.classification
        }
    elif api_name == "classify":
        # Read and encode all required files
        element_img = image_to_base64(args.element_image)
        full_img = image_to_base64(args.fullpage_image)
        dom_html = read_html_file(args.dom_html)

        return {
            "elementImage": element_img,
            "fullPageImage": full_img,  # Pass as base64 string
            "domHTML": dom_html,
            "xpath": args.xpath,
            "message": args.message or ""  # Default to empty string if not provided
        }
    elif api_name == "train":
        return {}  # Assuming no params needed
    else:
        print(f"Unsupported API: {api_name}")
        sys.exit(1)

def get_method(api_name):
    """Return the HTTP method for the API."""
    return "GET" if "predict" in api_name else "POST"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger ML API")

    parser.add_argument("--api", required=True, help="API name (e.g., predict, learnWithImage, classify)")
    parser.add_argument("--base_url", default="http://localhost:5000", help="Base URL of the API server")

    # Common optional arguments
    parser.add_argument("--message", help="Error or user message")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--classification", help="Classification label")
    parser.add_argument("--element_image", help="Path to element screenshot")
    parser.add_argument("--fullpage_image", help="Path to full page screenshot")
    parser.add_argument("--dom_html", help="Path to DOM HTML file")
    parser.add_argument("--xpath", help="Element XPath")

    args = parser.parse_args()
    api_name = args.api
    base_url = args.base_url
    method = get_method(api_name)
    params = get_params(api_name, args)

    result = call_api(api_name, base_url, method, params)
    print(json.dumps(result, indent=2))
