To install all the dependencies, run the following command:
```
pip install -r requirements.txt
```
## Usage
To run the application, run the following command:
```
python score_keywords.py
```

If 'python' does not work, try 'python3' instead.
```
python3 score_keywords.py
```

## Environment Variables

Script takes the hugging face api key as environment variable. 
To set the api key in WINDOWS, run the following command:
```
setx HUGGINGFACE_API_KEY <your-api-key> /m
```
To set the api key in MACOS, run the following command:
```
echo 'export HUGGINGFACE_API_KEY=<your-api-key>' >> ~/.zshrc
source ~/.zshrc
```