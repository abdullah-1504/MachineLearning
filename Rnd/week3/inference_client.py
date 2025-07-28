import requests
import argparse
import json
from pydantic import BaseModel, ValidationError, conint, confloat
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



class InferencePayload(BaseModel):
    Gender: conint(ge=0, le=1)
    Age: confloat(gt=0)
    Annual_Income_k: confloat(gt=0)
    Spending_Score: confloat(ge=0, le=100)


print("🟢 Inference script started...")


# Setup Argument Parser
parser = argparse.ArgumentParser(description="Customer Segmentation CLI Inference")
parser.add_argument('--gender', type=int, required=True, help='1 = Male, 0 = Female')
parser.add_argument('--age', type=float, required=True, help='Age of the customer')
parser.add_argument('--income', type=float, required=True, help='Annual income in k$')
parser.add_argument('--score', type=float, required=True, help='Spending score (1-100)')
parser.add_argument('--endpoint', choices=['cluster', 'segment'], default='cluster', help='Which API endpoint to hit')
args = parser.parse_args()

#validate payloads with pydantic
#This will raise an error if the input is invalid



#Prepare payload
try:    
    payload =InferencePayload( 
        Gender=args.gender,
        Age=args.age,
        Annual_Income_k=args.income,
        Spending_Score=args.score
).dict()

except ValidationError as e:
    print("❌ Invalid input data:", e)
    exit(1)

# Setup retry strategy for requests
retry_strategy = Retry(
    total=3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
    backoff_factor=1,
)

adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)


# Define endpoint
url = f"http://127.0.0.1:8000/predict-{args.endpoint}/"

print(" Hitting URL:", url)
print(" Payload:", payload)


#Make the POST request
try:
    response = requests.post(url, json=payload)
except requests.exceptions.RequestException as e:
    print("❌ Error during request:", e)
    exit()

#Output response
if response.status_code == 200:
    print("✅ Prediction Result:", response.json())
else:
    print("❌ Failed to get prediction:", response.status_code, response.text)
