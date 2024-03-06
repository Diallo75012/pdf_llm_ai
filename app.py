import gradio as gr



"""
### BLOCK EXAMPLE
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Say Hello")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
"""

"""
#### Different imput types dropdownm radio, slider. checkbox
def sentence_builder(quantity, animal, countries, place, activity_list, morning):
    return f"The {quantity} {animal}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"


demo = gr.Interface(
    sentence_builder,
    [
        gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
        gr.Dropdown(
            ["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"
        ),
        gr.CheckboxGroup(["USA", "Japan", "Pakistan"], label="Countries", info="Where are they from?"),
        gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?"),
        gr.Dropdown(
            ["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity", info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
        ),
        gr.Checkbox(label="Morning", info="Did they do it in the morning?"),
    ],
    "text",
    examples=[
        [2, "cat", ["Japan", "Pakistan"], "park", ["ate", "swam"], True],
        [4, "dog", ["Japan"], "zoo", ["ate", "swam"], False],
        [10, "bird", ["USA", "Pakistan"], "road", ["ran"], False],
        [8, "cat", ["Pakistan"], "zoo", ["ate"], True],
    ]
)


#### FUNCTION : HERE TEXT DIFFERENCE
from difflib import Differ

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


demo = gr.Interface(
    diff_texts,
    [
        gr.Textbox(
            label="Text 1",
            info="Initial text",
            lines=3,
            value="The quick brown fox jumped over the lazy dogs.",
        ),
        gr.Textbox(
            label="Text 2",
            info="Text to compare",
            lines=3,
            value="The fast brown fox jumps over lazy dogs.",
        ),
    ],
    gr.HighlightedText(
        label="Diff",
        combine_adjacent=True,
        show_legend=True,
        color_map={"+": "red", "-": "green"}),
    theme=gr.themes.Base()
)

#### TAB INTERFACE (Need for file uploaded details)
tts_examples = [
    "I love learning machine learning",
    "How do you do?",
]

tts_demo = gr.load(
    "huggingface/facebook/fastspeech2-en-ljspeech",
    title=None,
    examples=tts_examples,
    description="Give me something to say!",
)

stt_demo = gr.load(
    "huggingface/facebook/wav2vec2-base-960h",
    title=None,
    inputs=gr.Microphone(type="filepath"),
    description="Let me try to guess what you're saying!",
)

testtab = gr.TabbedInterface([tts_demo, stt_demo], ["Text-to-speech", "Speech-to-text"])
"""


###### DJANGO APP CUSTOM GRADIO FRONTEND ######

### VARS
pdf_files = []


### HELPER FUNCTIONS
def greet(name):
    return "Hello " + name + "!"

def print_file(file_path):
  with open(f"{file_path}", "r") as f:
    return f.read()

def chat_test(city, country):
    return f"{city}-{country}"

### EXAMPLES TO RUN FOR USERS
tts_examples = [
    "I love learning machine learning",
    "How do you do?",
]

### TAB LOADED COMPONENTS

# TTS TAB (Using Huggingface model): text imput, output voice audio file gr.load to get model from hugginface space or repo
tts_demo = gr.load(
    "huggingface/facebook/fastspeech2-en-ljspeech",
    title="Create Audio File",
    examples=tts_examples,
    description="Give me something to say!",
)

# STT TAB (Using Hugginggace model): record input microphone voice, output text
stt_demo = gr.load(
    "huggingface/facebook/wav2vec2-base-960h",
    title="Extract Text From Recorded Voice",
    inputs=gr.Microphone(type="filepath"),
    description="Let me try to guess what you're saying!",
)

# FILE TAB
# one file
doc_upload = gr.File(
    file_count = "single", # "mulitple", "directory", "single"
    file_types = [".pdf", ".txt", ".json"], # also to only have one type of file: "audio", "image", "video", "text"
)
print("doc_uploaded")
# several files
docs_upload  = gr.Files(
    file_count = "multiple",
    file_types = [".pdf", ".txt", ".json"],
)


# TEXT TAB
with gr.Blocks() as question_pdf:
    # custom title and text
    gr.Markdown(
    """
    # Talk To Your PDF
    
    Upload your file and ask me (-_-)/
    """)
    
    # upload file part
    gr.File(
      file_count = "single", # "mulitple", "directory", "single"
      file_types = [".pdf", ".txt", ".json"], # also to only have one type of file: "audio", "image", "video", "text"
    )
    gr.Interface(
      print_file,
      inputs=gr.FileExplorer(
        file_count="single",
        
      ),
      outputs="textbox"
    )
    
    # asking question to file part
    gr.ChatInterface(chat_test, inputs=["Tokyo","Japan"], outputs="text")
    
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Response") # use here the returned response from business logic function ()
    greet_btn = gr.Button("Ask Me")
    title="Upload PDF And Ask",
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet",)


# GROUP COMPONENT TO BE LAUNCHED IN WEBUI
testtab = gr.TabbedInterface([ question_pdf, tts_demo, stt_demo,], ["File-insights", "Text-to-speech", "Speech-to-text", ],)


import gradio as gr
from pathlib import Path

current_file_path = Path(__file__).resolve()
relative_path = "path/to/file"
absolute_path = (current_file_path.parent / ".." / ".." / "gradio").resolve()


def get_file_content(file):
    return (file,)


with gr.Blocks() as demo:
    gr.Markdown('### `FileExplorer` to `FileExplorer` -- `file_count="multiple"`')
    submit_btn = gr.Button("Select")
    with gr.Row():
        file = gr.FileExplorer(
            glob="*.py",
            # value=["themes/utils"],
            root=absolute_path,
            ignore_glob="**/__init__.py",
        )

        file2 = gr.FileExplorer(
            glob="*.py",
            root=absolute_path,
            ignore_glob="**/__init__.py",
        )
    submit_btn.click(lambda x: x, file, file2)

    gr.Markdown("---")
    gr.Markdown('### `FileExplorer` to `Code` -- `file_count="single"`')
    with gr.Group():
        with gr.Row():
            file_3 = gr.FileExplorer(
                scale=1,
                glob="*.py",
                value=["themes/utils"],
                file_count="single",
                root=absolute_path,
                ignore_glob="**/__init__.py",
                elem_id="file",
            )

            code = gr.Code(lines=30, scale=2, language="python")

    file_3.change(get_file_content, file_3, code)



if __name__ == "__main__":
    # demo.launch(debug=True)
    testtab.launch(
     debug=True,
     # not recommended to setup ssl like that but better have a reverse proxy and setup the ssl terminaison there
     ssl_keyfile="gradio-selfsigned.key",
     ssl_certfile="gradio-selfsigned.crt",
     ssl_verify=False,
    )
    
    
    
    
    
    
