# download model
mkdir models
wget https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf -P models/ 
# install dependencies
pip install --upgrade --force-reinstall -r requirements.txt
# start app and tunnel
uvicorn main:app --reload & npx localtunnel -p 8000