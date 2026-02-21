import requests
import json

try:
    response = requests.get("http://localhost:8000/api/ai/summaries")
    if response.status_code == 200:
        data = response.json()
        print("✅ API /api/ai/summaries is working")
        print(f"Summaries count: {len(data.get('summaries', []))}")
        if data.get('summaries'):
            print("Sample summary:", json.dumps(data['summaries'][0], indent=2))
    else:
        print(f"❌ API returned status {response.status_code}")
except Exception as e:
    print(f"❌ Error connecting to API: {e}")
