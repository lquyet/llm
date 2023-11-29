# download model
mkdir models
wget https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf -P models/ 
# install dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# compile and install llm-cpp-python for cuda if available
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama_cpp_python==0.2.20 --no-cache-dir
# else just compile the cpu version
#pip install llama_cpp_python==0.2.20

# start app and tunnel
uvicorn main:app --reload & npx localtunnel -p 8000
