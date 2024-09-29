import requests
import json

class OllamaModel:
    def __init__(self, model, temperature=0):
        """
        Initializes the OllamaModel with the given parameters.

        Parameters:
        model (str): The name of the model to use.
        system_prompt (str): The system prompt to use.
        temperature (float): The temperature setting for the model.
        stop (str): The stop token for the model.
        """
        self.model_endpoint = "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    def generate_text(self, prompt, system_prompt):
        """
        Generates a response from the Ollama model based on the provided prompt.

        Parameters:
        prompt (str): The user query to generate a response for.
        system_prompt (str): The system prompt to use.

        Returns:
        dict: The response from the model as a dictionary.
        """
        payload = {
            "model": self.model,
            "format": "json",
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "temperature": self.temperature,
        }

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            
            if request_response.status_code != 200:
                raise requests.RequestException(f"API request failed with status code: {request_response.status_code}")
            
            request_response_json = request_response.json()
            response = request_response_json.get('response')
            
            if not response:
                raise ValueError("No 'response' key in the API response")
            
            
            response_dict = json.loads(response)
            
            print(f"\n\nResponse from Ollama model: {response_dict}")
            
            return response_dict
        except requests.RequestException as e:
            print(f"RequestException: {str(e)}")
            response = {"error": f"Error in invoking model! {str(e)}"}
            return response
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {str(e)}")
            response = {"error": f"Error decoding JSON response! {str(e)}"}
            return response
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            response = {"error": f"Unexpected error occurred! {str(e)}"}
            return response
