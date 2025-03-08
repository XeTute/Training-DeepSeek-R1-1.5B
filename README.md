> [!NOTE]
> Example dataset `data.json` was downloaded from [here](https://huggingface.co/datasets/practical-dreamer/RPGPT_PublicDomain-alpaca/blob/main/RPGPT_PublicDomain_v1-alpaca.json).

# Training-DeepSeek-R1-1.5B
Minimal script I'm using to train DeepSeek R1 1.5B on a RTX4060 8GB

## 🏃‍♂️ Quickstart
1. Install libraries (prefer venv):
  ```cmd
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # Or your hardware-specific torch install command
  pip install transformers datasets
  ```
2. Clone & enter this repo:
  ```cmd
  git clone https://github.com/XeTute/Training-DeepSeek-R1-1.5B
  cd Training-DeepSeek-R1-1.5B
  ```
3. Verify if JSON data is valid through `python verify.py`
4. Train through `python main.py` =)

## Customize

### Customize Training Arguments
Line six to line ten look like following:
```py
# constants for you to set
epochs = 1 # more takes more
batchsize = 1 # per device
gradientaccum = 4 # if your context length is already too low, prefer lowering this
maxctxlen = 4 * 1024
```
Feel free to modify these parameters given how much compute, memory, et cetera you got

### Customize Data to train on

> [!TIP]
> Don't have any data for your use-case? Synthetically generate it using [this script](https://github.com/XeTute/Synthetic-Data-Generation) 

To modify the data you'll be training on, make sure it follows this JSON format (known as Alpaca I believe):
```JSON
[
  {
    "instruction": "System prompt comes here",
    "input": "user input comes here",
    "output": "expected output comes here"
  },
  repeat instruction input output inside {}
]
```
After swapping the data, you can verify if it's valid JSON or not by running `python verify.py` and reading the output.

---
# Our Apps & Socials
[Chat with our Assistant](https://xetute.com/) | [Support us Financially](https://ko-fi.com/XeTute) | [Visit our GitHub](https://github.com/XeTute)  

Long live the Islamic Republic of Pakistan; Glory to the Islamic Republic of Pakistan 🇵🇰  
![The Flag of the Islamic Federal Republic of Pakistan](https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg)
