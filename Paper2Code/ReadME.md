In this file, we are trying to reproduce the process of paper "Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning" by using "All You Need is Attention" as an example.  
Link of "Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning" : https://github.com/going-doer/Paper2Code?tab=readme-ov-file



Detailed Methods on bash:   
1. git clone https://github.com/going-doer/Paper2Code.git
2. cd Paper2Code
3. pip install -r requirements.txt
4. export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
5. cd scripts
6. bash run.sh
7. cd ../outputs/Transformer_repo
8. python main.py   
