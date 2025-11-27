import logging
import json
import requests
import os
import azure.functions as func

# Get config from Application Settings
AZURE_ML_ENDPOINT = os.environ.get(
    'AZURE_ML_ENDPOINT',
    'https://sales-sa-ml-gdjrr.southeastasia.inference.ml.azure.com/score'
)
AZURE_ML_API_KEY = os.environ.get('AZURE_ML_API_KEY', '')

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Prediction request received')
    
    # Handle CORS preflight
    if req.method == 'OPTIONS':
        return func.HttpResponse(
            status_code=200,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
    
    try:
        # Get request body
        req_body = req.get_json()
        logging.info(f'Request data: {json.dumps(req_body)}')
        
        # Forward to Azure ML
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AZURE_ML_API_KEY}'
        }
        
        response = requests.post(
            AZURE_ML_ENDPOINT,
            headers=headers,
            json=req_body,
            timeout=30
        )
        
        logging.info(f'Azure ML response status: {response.status_code}')
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f'Azure ML raw result: {result}')
            
            # Azure ML may return string containing JSON, parse it
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                    logging.info(f'Parsed result: {result}')
                except json.JSONDecodeError:
                    logging.warning('Could not parse result as JSON')
            
            return func.HttpResponse(
                json.dumps(result),
                status_code=200,
                headers={
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        else:
            error_msg = response.text
            logging.error(f'Azure ML error: {error_msg}')
            
            return func.HttpResponse(
                json.dumps({
                    'error': f'Azure ML returned {response.status_code}',
                    'details': error_msg
                }),
                status_code=response.status_code,
                headers={
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            )
    
    except ValueError:
        return func.HttpResponse(
            json.dumps({'error': 'Invalid JSON in request body'}),
            status_code=400,
            headers={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        
        return func.HttpResponse(
            json.dumps({
                'error': 'Internal server error',
                'details': str(e)
            }),
            status_code=500,
            headers={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
